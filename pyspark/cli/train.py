"""
Pipeline d'entraînement du modèle de churn.

Ce script enchaîne :
  [1/4] Chargement des features (Parquet ou rebuild depuis CSV)
  [2/4] Split train / test
  [3/4] Entraînement du modèle (Random Forest | GBT | Logistic)
  [4/4] Évaluation et sauvegarde du modèle

Usage :
    python -m cli.train
    python -m cli.train --features data/processed/features
    python -m cli.train --algo gbt --num-trees 100 --max-depth 10
    python -m cli.train --cv                     # avec cross-validation
    python -m cli.train --env production
    python -m cli.train --cutoff 2011-09-01      # rebuild features au vol

Modèle sauvegardé dans : output/model/
Métriques JSON dans    : output/metrics.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _banner(title: str) -> None:
    print()
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)


def _step(n: int, total: int, label: str) -> None:
    print()
    print(f"── [{n}/{total}] {label} {'─' * (52 - len(label))}")


def _ok(msg: str) -> None:
    print(f" {msg}")


def _info(msg: str) -> None:
    print(f" [info] {msg}")


# ──────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Entraînement du modèle de churn.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--env", default="local",
                   help="Environnement config (local | production). Défaut: local.")
    p.add_argument("--features", default=None,
                   help="Chemin Parquet des features (défaut: data/processed/features).")
    p.add_argument("--csv", default=None,
                   help="CSV brut (si les features ne sont pas disponibles).")
    p.add_argument("--cutoff", default=None,
                   help="Date cutoff RFM (YYYY-MM-DD) — utilisée si rebuild des features.")
    p.add_argument("--output-dir", default="output",
                   help="Dossier de sortie pour le modèle et les métriques. Défaut: output.")
    # Hyperparamètres
    p.add_argument("--algo", default=None,
                   choices=["random_forest", "gbt", "logistic"],
                   help="Algorithme ML. Défaut: depuis config (random_forest).")
    p.add_argument("--num-trees", type=int, default=None,
                   help="Nombre d'arbres (RF/GBT). Défaut: depuis config (50).")
    p.add_argument("--max-depth", type=int, default=None,
                   help="Profondeur max (RF/GBT). Défaut: depuis config (8).")
    p.add_argument("--train-ratio", type=float, default=None,
                   help="Ratio train/test (0–1). Défaut: 0.8.")
    p.add_argument("--cv", action="store_true",
                   help="Activer la cross-validation 3 plis.")
    p.add_argument("--seed", type=int, default=None,
                   help="Graine aléatoire. Défaut: 42.")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    t0_global = time.time()

    # ── Config ────────────────────────────────────────────────────────────────
    from src.config import load_config
    cfg = load_config(args.env)

    # Overrides CLI → config
    if args.cutoff:
        cfg.features.cutoff_date = args.cutoff
    if args.algo:
        cfg.model.algorithm = args.algo
    if args.num_trees:
        cfg.model.num_trees = args.num_trees
    if args.max_depth:
        cfg.model.max_depth = args.max_depth
    if args.train_ratio:
        cfg.model.train_ratio = args.train_ratio
    if args.cv:
        cfg.model.use_cv = True
    if args.seed:
        cfg.model.seed = args.seed

    features_path = Path(args.features) if args.features else Path(cfg.data.processed_path) / "features"
    output_dir    = Path(args.output_dir)
    model_dir     = output_dir / "model"
    metrics_path  = output_dir / "metrics.json"

    _banner("Pipeline Entraînement — Churn Prediction")
    _info(f"Environnement  : {args.env}")
    _info(f"Features       : {features_path}")
    _info(f"Algorithme     : {cfg.model.algorithm}")
    _info(f"Num trees      : {cfg.model.num_trees}")
    _info(f"Max depth      : {cfg.model.max_depth}")
    _info(f"Cross-val      : {cfg.model.use_cv}")
    _info(f"Train ratio    : {cfg.model.train_ratio}")
    _info(f"Modèle out     : {model_dir}")

    # ── SparkSession ──────────────────────────────────────────────────────────
    from src.spark.session import get_spark
    spark = get_spark(cfg.spark, app_name="TrainPipeline")
    _ok(f"SparkSession démarrée ({cfg.spark.master})")

    # ══════════════════════════════════════════════════════════════════════════
    # ÉTAPE 1 — Chargement des features
    # ══════════════════════════════════════════════════════════════════════════
    _step(1, 4, "Chargement des features")
    t0 = time.time()

    if features_path.exists():
        _info(f"Lecture Parquet : {features_path}")
        features_df = spark.read.parquet(str(features_path))
    else:
        _info(f"Parquet introuvable ({features_path})")
        _info("Rebuild des features depuis le CSV...")

        from src.data.download import load_transactions
        from src.data.cleaning import clean_transactions
        from src.feature.builder import build_features_and_label
        from datetime import datetime

        raw_df = load_transactions(spark, cfg.data, csv_path=args.csv)
        clean_df = clean_transactions(raw_df)
        cutoff_dt = datetime.strptime(cfg.features.cutoff_date, "%Y-%m-%d")
        features_df = build_features_and_label(
            clean_df, cutoff_dt, cfg.features.horizon_days
        )
        _info("Pour éviter ce rebuild, lancez d'abord : python -m cli.data_download")

    n_total = features_df.cache().count()
    _ok(f"{n_total:,} clients chargés")
    _info(f"Colonnes : {features_df.columns}")
    _info(f"Durée étape 1 : {time.time() - t0:.1f}s")

    # ══════════════════════════════════════════════════════════════════════════
    # ÉTAPE 2 — Split train / test
    # ══════════════════════════════════════════════════════════════════════════
    _step(2, 4, "Split train / test")
    t0 = time.time()

    from src.models.train import split_data
    train_df, test_df = split_data(features_df, cfg.model.train_ratio, cfg.model.seed)

    n_train = train_df.cache().count()
    n_test  = test_df.cache().count()
    _ok(f"Train : {n_train:,}  |  Test : {n_test:,}  ({cfg.model.train_ratio*100:.0f}/{(1-cfg.model.train_ratio)*100:.0f})")
    _info(f"Durée étape 2 : {time.time() - t0:.1f}s")

    # ══════════════════════════════════════════════════════════════════════════
    # ÉTAPE 3 — Entraînement
    # ══════════════════════════════════════════════════════════════════════════
    _step(3, 4, f"Entraînement ({cfg.model.algorithm}{'  + CV 3-plis' if cfg.model.use_cv else ''})")
    t0 = time.time()

    from src.models.train import train
    model = train(train_df, cfg.model, cfg.features.feature_cols)

    elapsed_train = time.time() - t0
    _ok(f"Modèle entraîné en {elapsed_train:.1f}s")

    # ══════════════════════════════════════════════════════════════════════════
    # ÉTAPE 4 — Évaluation + sauvegarde
    # ══════════════════════════════════════════════════════════════════════════
    _step(4, 4, "Évaluation et sauvegarde")
    t0 = time.time()

    from src.models.evaluate import evaluate, print_evaluation
    metrics = evaluate(model, test_df, cfg.features.feature_cols)
    print_evaluation(metrics)

    # Sauvegarde modèle
    model_dir.mkdir(parents=True, exist_ok=True)
    model.write().overwrite().save(str(model_dir))
    _ok(f"Modèle sauvegardé : {model_dir}")

    # Sauvegarde métriques JSON
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_serializable = {
        k: (
            {str(kk): vv for kk, vv in v.items()}  # confusion matrix keys
            if isinstance(v, dict) else v
        )
        for k, v in metrics.items()
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_serializable, f, indent=2, ensure_ascii=False)
    _ok(f"Métriques JSON   : {metrics_path}")
    _info(f"Durée étape 4 : {time.time() - t0:.1f}s")

    # ── Résumé final ──────────────────────────────────────────────────────────
    elapsed = time.time() - t0_global
    _banner(f"✅ Entraînement terminé en {elapsed:.0f}s")
    print(f"   AUC       : {metrics['auc']}")
    print(f"   F1 score  : {metrics['f1']}")
    print(f"   Accuracy  : {metrics['accuracy']}")
    print(f"   Modèle    : {model_dir}")
    print(f"   Métriques : {metrics_path}")
    print()
    print("   Prochaine étape :")
    print(f"     python -m cli.predict --model-dir {model_dir}")
    print()

    spark.stop()


if __name__ == "__main__":
    main()
