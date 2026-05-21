"""
Pipeline ML complet — données, entraînement, scoring et monitoring.

Ce script est le point d'entrée unique qui enchaîne les 4 étapes :
  [1/4] DataPipeline   → téléchargement UCI + nettoyage + features RFM
  [2/4] Training       → split train/test + entraînement + évaluation
  [3/4] Scoring        → scoring de tous les clients + top-N à risque
  [4/4] Monitoring     → détection de drift PSI (features + scores)

Artefacts produits (dans --output-dir, défaut: output/) :
  output/
  ├── model/                  ← PipelineModel PySpark
  ├── metrics.json            ← AUC, F1, accuracy, confusion matrix
  ├── predictions.csv         ← tous les clients scorés (trié par proba)
  ├── top_at_risk.csv         ← top-N clients les plus à risque
  ├── pipeline_run.json       ← résumé de toutes les étapes
  └── monitoring/
      └── drift_report.json   ← rapport PSI par feature + score

Usage :
    python -m cli.pipeline
    python -m cli.pipeline --skip-download   # CSV déjà présent
    python -m cli.pipeline --skip-train      # modèle déjà entraîné
    python -m cli.pipeline --force-download  # re-télécharge tout
    python -m cli.pipeline --env production --algo gbt --cv
    python -m cli.pipeline --cutoff 2011-09-01 --horizon-days 90
    python -m cli.pipeline --no-monitoring   # sans étape PSI
    python -m cli.pipeline --psi-warning 0.05 --psi-critical 0.15
"""
import argparse
import sys
import time
from pathlib import Path


# ──────────────────────────────────────────────────────────────────────────────
# Helpers console
# ──────────────────────────────────────────────────────────────────────────────

def _banner(title: str) -> None:
    print()
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)


def _section(label: str) -> None:
    print(f"\n  {'─' * 4}  {label}")


# ──────────────────────────────────────────────────────────────────────────────
# Parsing des arguments
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Pipeline ML complet : données → entraînement → scoring → monitoring.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── Environnement ──────────────────────────────────────────────────────────
    env = p.add_argument_group("Environnement")
    env.add_argument("--env", default="local",
                     help="Config environment (local | production). Défaut: local.")
    env.add_argument("--output-dir", default="output",
                     help="Dossier de sortie racine. Défaut: output.")

    # ── Données ───────────────────────────────────────────────────────────────
    data = p.add_argument_group("Données")
    data.add_argument("--csv", default=None,
                      help="Chemin CSV brut alternatif (override téléchargement).")
    data.add_argument("--cutoff", default=None,
                      help="Date de cutoff RFM (YYYY-MM-DD). Override config.")
    data.add_argument("--horizon-days", type=int, default=None,
                      help="Horizon churn en jours. Override config (défaut: 90).")
    data.add_argument("--skip-download", action="store_true",
                      help="Ne pas télécharger si le CSV existe déjà.")
    data.add_argument("--force-download", action="store_true",
                      help="Forcer le re-téléchargement même si le CSV est présent.")

    # ── Entraînement ──────────────────────────────────────────────────────────
    train = p.add_argument_group("Entraînement")
    train.add_argument("--skip-train", action="store_true",
                       help="Charger le modèle existant (output/model) sans ré-entraîner.")
    train.add_argument("--algo", default=None,
                       choices=["random_forest", "gbt", "logistic"],
                       help="Algorithme ML. Override config (défaut: random_forest).")
    train.add_argument("--num-trees", type=int, default=None,
                       help="Nombre d'arbres RF/GBT. Override config.")
    train.add_argument("--max-depth", type=int, default=None,
                       help="Profondeur max RF/GBT. Override config.")
    train.add_argument("--cv", action="store_true",
                       help="Activer la cross-validation 3 plis.")

    # ── Scoring ───────────────────────────────────────────────────────────────
    score = p.add_argument_group("Scoring")
    score.add_argument("--threshold", type=float, default=0.5,
                       help="Seuil de classification churn [0–1]. Défaut: 0.5.")
    score.add_argument("--top-n", type=int, default=20,
                       help="Nb de clients dans top_at_risk.csv. Défaut: 20.")

    # ── Monitoring ────────────────────────────────────────────────────────────
    mon = p.add_argument_group("Monitoring")
    mon.add_argument("--no-monitoring", action="store_true",
                     help="Désactiver l'étape de monitoring PSI.")
    mon.add_argument("--psi-warning", type=float, default=None,
                     help="Seuil PSI WARNING. Override config (défaut: 0.10).")
    mon.add_argument("--psi-critical", type=float, default=None,
                     help="Seuil PSI CRITICAL. Override config (défaut: 0.25).")
    mon.add_argument("--n-bins", type=int, default=None,
                     help="Nombre de bins pour le calcul PSI. Défaut: 10.")

    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    t0   = time.perf_counter()

    # ── Config ────────────────────────────────────────────────────────────────
    from src.config import load_config
    cfg = load_config(args.env)

    # Overrides CLI → config
    if args.cutoff:       cfg.features.cutoff_date     = args.cutoff
    if args.horizon_days: cfg.features.horizon_days     = args.horizon_days
    if args.algo:         cfg.model.algorithm           = args.algo
    if args.num_trees:    cfg.model.num_trees           = args.num_trees
    if args.max_depth:    cfg.model.max_depth           = args.max_depth
    if args.cv:           cfg.model.use_cv              = True
    if args.psi_warning:  cfg.monitoring.psi_threshold_warning  = args.psi_warning
    if args.psi_critical: cfg.monitoring.psi_threshold_critical = args.psi_critical
    if args.n_bins:       cfg.monitoring.n_bins         = args.n_bins

    # ── Bannière ──────────────────────────────────────────────────────────────
    total_steps = 4 if not args.no_monitoring else 3
    _banner(f"Pipeline ML Complet  ({total_steps} étapes)")

    _section("Configuration")
    print(f"    Environnement  : {args.env}")
    print(f"    Cutoff RFM     : {cfg.features.cutoff_date}")
    print(f"    Horizon churn  : {cfg.features.horizon_days} jours")
    print(f"    Algorithme     : {cfg.model.algorithm}")
    print(f"    Cross-val      : {cfg.model.use_cv}")
    print(f"    Seuil churn    : {args.threshold}")
    print(f"    Seuils PSI     : WARNING={cfg.monitoring.psi_threshold_warning}"
          f"  CRITICAL={cfg.monitoring.psi_threshold_critical}")
    print(f"    Monitoring     : {'désactivé (--no-monitoring)' if args.no_monitoring else 'actif'}")
    print(f"    Output dir     : {args.output_dir}")

    # ── SparkSession ──────────────────────────────────────────────────────────
    _section("Spark")
    from src.spark.session import get_spark
    spark = get_spark(cfg.spark, app_name="FullPipeline")
    print(f"    SparkSession démarrée  ({cfg.spark.master})")
    print(f"    Driver memory          : {cfg.spark.driver_memory}")

    # ── Orchestration ─────────────────────────────────────────────────────────
    from src.orchestration.pipeline import run_full_pipeline

    pipeline_result = run_full_pipeline(
        spark          = spark,
        cfg            = cfg,
        csv_path       = args.csv,
        output_dir     = args.output_dir,
        skip_download  = args.skip_download,
        force_download = args.force_download,
        skip_train     = args.skip_train,
        threshold      = args.threshold,
        top_n          = args.top_n,
        run_monitoring = not args.no_monitoring,
    )

    # ── Récapitulatif des fichiers produits ───────────────────────────────────
    out = Path(args.output_dir)
    _banner("Fichiers produits")

    artifacts = [
        ("Features Parquet",    out.parent / "data" / "processed" / "features"),
        ("Modèle PySpark",      out / "model"),
        ("Métriques JSON",      out / "metrics.json"),
        ("Prédictions CSV",     out / "predictions.csv"),
        (f"Top-{args.top_n} à risque", out / "top_at_risk.csv"),
        ("Résumé pipeline",     out / "pipeline_run.json"),
        ("Rapport drift PSI",   out / "monitoring" / "drift_report.json"),
    ]

    for label, path in artifacts:
        exists = "✅" if path.exists() else "  "
        print(f"  {exists} {label:<25} {path}")

    # ── Prochaines étapes ─────────────────────────────────────────────────────
    elapsed = time.perf_counter() - t0
    print()
    print(f"  ⏱  Durée totale : {elapsed:.0f}s")
    print()
    print("  Prochaines étapes :")
    print(f"    python -m cli.serve                     # lancer l'API FastAPI")
    print(f"    python -m cli.pipeline --skip-download  # ré-entraîner sans re-télécharger")
    print(f"    python -m cli.pipeline --skip-train     # scoring + monitoring seuls")
    print()

    spark.stop()
    sys.exit(0 if pipeline_result.success else 1)


if __name__ == "__main__":
    main()
