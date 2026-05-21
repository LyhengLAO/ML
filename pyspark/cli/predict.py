"""
Pipeline de prédiction churn — scoring d'un dataset de clients.

Ce script applique un modèle entraîné sur des features de clients
et produit les scores de risque de churn.

Étapes :
  [1/3] Chargement du modèle et des features
  [2/3] Scoring (prédiction + probabilité de churn)
  [3/3] Affichage du Top-N clients à risque + sauvegarde CSV

Usage :
    python -m cli.predict
    python -m cli.predict --model-dir output/model
    python -m cli.predict --model-dir output/model --top-n 50
    python -m cli.predict --features data/processed/features --output output/predictions.csv
    python -m cli.predict --env production

Sorties :
    Affichage console  : Top-N clients à risque (CustomerID + probabilité)
    Fichier CSV        : output/predictions.csv (tous les clients scorés)
"""
import argparse
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
    print(f"  {msg}")


def _info(msg: str) -> None:
    print(f" [info] {msg}")


def _warn(msg: str) -> None:
    print(f" [WARNING]  {msg}")


# ──────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Prédiction churn sur un jeu de features.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--env", default="local",
                   help="Environnement config (local | production). Défaut: local.")
    p.add_argument("--model-dir", default=None,
                   help="Dossier du modèle PipelineModel (défaut: output/model).")
    p.add_argument("--features", default=None,
                   help="Chemin Parquet des features (défaut: data/processed/features).")
    p.add_argument("--csv", default=None,
                   help="CSV brut alternatif si le Parquet des features est absent.")
    p.add_argument("--cutoff", default=None,
                   help="Date cutoff (YYYY-MM-DD) pour rebuild features si besoin.")
    p.add_argument("--output", default=None,
                   help="Fichier CSV de sortie pour toutes les prédictions. "
                        "Défaut: output/predictions.csv.")
    p.add_argument("--top-n", type=int, default=20,
                   help="Nombre de clients à risque affichés en console. Défaut: 20.")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="Seuil de classification churn (0–1). Défaut: 0.5.")
    p.add_argument("--no-save", action="store_true",
                   help="Ne pas sauvegarder le CSV de prédictions.")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Affichage console du Top-N
# ──────────────────────────────────────────────────────────────────────────────

def _print_top_n(top_rows: list, n: int) -> None:
    """Affiche un tableau formaté des N clients les plus à risque."""
    print()
    print(f"  {'#':<5} {'CustomerID':<14} {'Churn Proba':>12} {'Prediction':>12}")
    print(f"  {'─'*5} {'─'*14} {'─'*12} {'─'*12}")
    for i, row in enumerate(top_rows, 1):
        pred_label = "🔴 CHURN" if row["prediction"] == 1 else "🟢 OK"
        print(f"  {i:<5} {row['CustomerID']:<14} "
              f"{row['churn_proba']:>11.4f}  {pred_label:>12}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    t0_global = time.time()

    # ── Config ────────────────────────────────────────────────────────────────
    from src.config import load_config
    cfg = load_config(args.env)

    if args.cutoff:
        cfg.features.cutoff_date = args.cutoff

    model_dir     = Path(args.model_dir) if args.model_dir else Path("output") / "model"
    features_path = Path(args.features)  if args.features  else Path(cfg.data.processed_path) / "features"
    output_csv    = Path(args.output)    if args.output    else Path("output") / "predictions.csv"

    _banner("Pipeline Prédiction — Churn Scoring")
    _info(f"Environnement  : {args.env}")
    _info(f"Modèle         : {model_dir}")
    _info(f"Features       : {features_path}")
    _info(f"Top-N affiché  : {args.top_n}")
    _info(f"Seuil churn    : {args.threshold}")
    _info(f"Sortie CSV     : {output_csv}")

    # ── Vérification modèle ───────────────────────────────────────────────────
    if not model_dir.exists():
        print(f"\n   ❌ Modèle introuvable : {model_dir}", file=sys.stderr)
        print("   Entraînez d'abord le modèle :", file=sys.stderr)
        print("       python -m cli.train", file=sys.stderr)
        sys.exit(1)

    # ── SparkSession ──────────────────────────────────────────────────────────
    from src.spark.session import get_spark
    spark = get_spark(cfg.spark, app_name="PredictPipeline")
    _ok(f"SparkSession démarrée ({cfg.spark.master})")

    # ══════════════════════════════════════════════════════════════════════════
    # ÉTAPE 1 — Chargement modèle + features
    # ══════════════════════════════════════════════════════════════════════════
    _step(1, 3, "Chargement du modèle et des features")
    t0 = time.time()

    # Chargement modèle
    from src.models.predict import load_model
    _info(f"Chargement du modèle PipelineModel depuis : {model_dir}")
    model = load_model(str(model_dir))
    _ok("Modèle chargé")

    # Chargement features
    if features_path.exists():
        _info(f"Lecture Parquet : {features_path}")
        features_df = spark.read.parquet(str(features_path))
    else:
        _warn(f"Parquet features introuvable ({features_path})")
        _info("Rebuild depuis CSV...")

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
        _info("Conseil : lancez d'abord python -m cli.data_download pour éviter ce rebuild.")

    n_clients = features_df.cache().count()
    _ok(f"{n_clients:,} clients à scorer")
    _info(f"Durée étape 1 : {time.time() - t0:.1f}s")

    # ══════════════════════════════════════════════════════════════════════════
    # ÉTAPE 2 — Scoring
    # ══════════════════════════════════════════════════════════════════════════
    _step(2, 3, "Scoring — calcul des probabilités de churn")
    t0 = time.time()

    from src.models.predict import predict_batch
    from pyspark.sql import functions as F

    scored_df = predict_batch(model, features_df)

    # Appliquer le seuil personnalisé si différent de 0.5
    if args.threshold != 0.5:
        scored_df = scored_df.withColumn(
            "prediction",
            F.when(F.col("churn_proba") >= args.threshold, 1).otherwise(0).cast("double")
        )
        _info(f"Seuil personnalisé appliqué : {args.threshold}")

    scored_df = scored_df.cache()
    n_scored  = scored_df.count()

    # Statistiques globales
    stats = (
        scored_df.agg(
            F.sum(F.col("prediction").cast("int")).alias("n_churn"),
            F.avg("churn_proba").alias("avg_proba"),
            F.max("churn_proba").alias("max_proba"),
        ).collect()[0]
    )
    n_churn   = stats["n_churn"] or 0
    n_retained = n_scored - n_churn

    _ok(f"{n_scored:,} clients scorés")
    print(f"     Prédits CHURN    : {n_churn:,}  ({n_churn/n_scored*100:.1f}%)")
    print(f"     Prédits RETENUS  : {n_retained:,}  ({n_retained/n_scored*100:.1f}%)")
    print(f"     Proba moy. churn : {stats['avg_proba']:.4f}")
    print(f"     Proba max churn  : {stats['max_proba']:.4f}")
    _info(f"Durée étape 2 : {time.time() - t0:.1f}s")

    # ══════════════════════════════════════════════════════════════════════════
    # ÉTAPE 3 — Affichage Top-N + sauvegarde
    # ══════════════════════════════════════════════════════════════════════════
    _step(3, 3, f"Top-{args.top_n} clients à risque + sauvegarde")
    t0 = time.time()

    from src.models.predict import top_n_at_risk
    top_df  = top_n_at_risk(scored_df, args.top_n)
    top_rows = top_df.collect()

    print()
    print(f"  ┌─ Top {args.top_n} clients les plus à risque de churn ─────────────────┐")
    _print_top_n(top_rows, args.top_n)
    print(f"  └{'─'*60}┘")

    # Sauvegarde CSV
    if not args.no_save:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        (
            scored_df
            .orderBy(F.col("churn_proba").desc())
            .toPandas()
            .to_csv(str(output_csv), index=False)
        )
        _ok(f"Prédictions sauvegardées ({n_scored:,} lignes) : {output_csv}")

    _info(f"Durée étape 3 : {time.time() - t0:.1f}s")

    # ── Résumé final ──────────────────────────────────────────────────────────
    elapsed = time.time() - t0_global
    _banner(f"✅ Prédictions terminées en {elapsed:.0f}s")
    print(f"   Clients scorés   : {n_scored:,}")
    print(f"   Prédits churnés  : {n_churn:,}  ({n_churn/n_scored*100:.1f}%)")
    if not args.no_save:
        print(f"   Fichier CSV      : {output_csv}")
    print()

    spark.stop()


if __name__ == "__main__":
    main()
