"""
Pipeline complet de données — téléchargement, nettoyage et feature engineering.

Ce script enchaîne toutes les étapes de préparation des données :
  [1/4] Téléchargement du dataset UCI Online Retail II (~45 MB)
  [2/4] Chargement du CSV brut dans Spark
  [3/4] Nettoyage des transactions (filtres qualité + rapport)
  [4/4] Construction des features RFM + label churn

Le résultat est sauvegardé en Parquet dans data/processed/features/
et peut être consommé directement par `cli/train.py`.

Usage :
    python -m cli.data_download
    python -m cli.data_download --cutoff 2011-09-01 --horizon-days 90
    python -m cli.data_download --skip-download          # si CSV déjà présent
    python -m cli.data_download --env production
    python -m cli.data_download --force                  # re-télécharge tout

Dépendances : pandas, openpyxl, pyspark, requests
"""
import argparse
import sys
import time
from datetime import datetime
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
        description="Pipeline données : téléchargement → nettoyage → features.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--env", default="local",
                   help="Environnement config (local | production). Défaut: local.")
    p.add_argument("--data-dir", default=None,
                   help="Dossier raw (défaut: data/raw depuis config).")
    p.add_argument("--output-dir", default=None,
                   help="Dossier features Parquet (défaut: data/processed/features).")
    p.add_argument("--cutoff", default=None,
                   help="Date de cutoff RFM (YYYY-MM-DD). Défaut: depuis config.")
    p.add_argument("--horizon-days", type=int, default=None,
                   help="Horizon churn en jours. Défaut: depuis config (90).")
    p.add_argument("--skip-download", action="store_true",
                   help="Ne pas re-télécharger si le CSV brut est déjà présent.")
    p.add_argument("--force", action="store_true",
                   help="Forcer le re-téléchargement même si le CSV existe.")
    p.add_argument("--keep-zip", action="store_true",
                   help="Conserver le ZIP et le dossier extrait après conversion.")
    p.add_argument("--no-quality-report", action="store_true",
                   help="Désactiver le rapport qualité avant/après nettoyage.")
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

    raw_dir    = Path(args.data_dir)    if args.data_dir    else Path(cfg.data.raw_path)
    output_dir = Path(args.output_dir)  if args.output_dir  else Path(cfg.data.processed_path) / "features"
    csv_path   = raw_dir / "online_retail_II.csv"

    cutoff_str   = args.cutoff       or cfg.features.cutoff_date
    horizon_days = args.horizon_days or cfg.features.horizon_days

    _banner("Pipeline Données — UCI Online Retail II")
    _info(f"Environnement  : {args.env}")
    _info(f"CSV source     : {csv_path}")
    _info(f"Features out   : {output_dir}")
    _info(f"Cutoff date    : {cutoff_str}")
    _info(f"Horizon churn  : {horizon_days} jours")

    # ══════════════════════════════════════════════════════════════════════════
    # ÉTAPE 1 — Téléchargement
    # ══════════════════════════════════════════════════════════════════════════
    _step(1, 4, "Téléchargement UCI Online Retail II")
    t0 = time.time()

    from src.data.download import zip_file_download, extract

    if csv_path.exists() and not args.force:
        if args.skip_download:
            _ok(f"CSV déjà présent : {csv_path}  (--skip-download actif)")
        else:
            _ok(f"CSV déjà présent : {csv_path}")
            _info("Utilisez --force pour re-télécharger.")
    else:
        _info(f"Téléchargement vers : {raw_dir}")
        zip_file_download(dest=raw_dir)
        print()
        _info("Extraction et conversion Excel → CSV...")
        extract()
        _ok(f"CSV sauvegardé : {csv_path}")

    _info(f"Durée étape 1 : {time.time() - t0:.1f}s")

    # ══════════════════════════════════════════════════════════════════════════
    # ÉTAPE 2 — Chargement CSV dans Spark
    # ══════════════════════════════════════════════════════════════════════════
    _step(2, 4, "Chargement CSV dans Spark")
    t0 = time.time()

    from src.spark.session import get_spark
    from src.data.download import load_csv

    spark = get_spark(cfg.spark, app_name="DataPipeline")
    _ok(f"SparkSession démarrée ({cfg.spark.master})")

    raw_df = load_csv(spark, str(csv_path))
    n_raw  = raw_df.cache().count()
    _ok(f"{n_raw:,} lignes chargées depuis le CSV")
    _info(f"Colonnes : {raw_df.columns}")
    _info(f"Durée étape 2 : {time.time() - t0:.1f}s")

    # ══════════════════════════════════════════════════════════════════════════
    # ÉTAPE 3 — Nettoyage des transactions
    # ══════════════════════════════════════════════════════════════════════════
    _step(3, 4, "Nettoyage des transactions")
    t0 = time.time()

    from src.data.cleaning import clean_transactions, quality_report

    if not args.no_quality_report:
        _info("Rapport qualité AVANT nettoyage :")
        report_before = quality_report(raw_df)
        print(f"     Lignes totales     : {report_before['total_rows']:,}")
        print(f"     Clients distincts  : {report_before['distinct_customers']:,}")
        nulls_before = {k: v for k, v in report_before["nulls_per_column"].items() if v and v > 0}
        if nulls_before:
            print(f"     Nulls par colonne  : {nulls_before}")

    clean_df  = clean_transactions(raw_df)
    n_clean   = clean_df.cache().count()
    n_dropped = n_raw - n_clean

    _ok(f"{n_clean:,} lignes après nettoyage (supprimé : {n_dropped:,})")

    if not args.no_quality_report:
        _info("Rapport qualité APRÈS nettoyage :")
        report_after = quality_report(clean_df)
        print(f"     Lignes totales     : {report_after['total_rows']:,}")
        print(f"     Clients distincts  : {report_after['distinct_customers']:,}")
        print(f"     Taux de rétention  : {n_clean / n_raw * 100:.1f}%")

    _info(f"Durée étape 3 : {time.time() - t0:.1f}s")

    # ══════════════════════════════════════════════════════════════════════════
    # ÉTAPE 4 — Feature engineering RFM + label
    # ══════════════════════════════════════════════════════════════════════════
    _step(4, 4, "Construction des features RFM + label churn")
    t0 = time.time()

    from src.feature.builder import build_features_and_label

    cutoff_dt = datetime.strptime(cutoff_str, "%Y-%m-%d")
    _info(f"Cutoff : {cutoff_dt.date()}  |  Horizon : {horizon_days} j")

    features_df = build_features_and_label(clean_df, cutoff_dt, horizon_days)
    n_clients   = features_df.cache().count()

    # Statistiques sur le label
    from pyspark.sql import functions as F
    label_dist = (
        features_df.groupBy("label")
                   .count()
                   .orderBy("label")
                   .collect()
    )
    churned   = next((r["count"] for r in label_dist if r["label"] == 1), 0)
    retained  = next((r["count"] for r in label_dist if r["label"] == 0), 0)

    _ok(f"{n_clients:,} clients dans le dataset final")
    print(f"     Churnés (label=1)  : {churned:,}  ({churned/n_clients*100:.1f}%)")
    print(f"     Retenus (label=0)  : {retained:,}  ({retained/n_clients*100:.1f}%)")
    print(f"     Features           : {[c for c in features_df.columns if c not in ('CustomerID','label')]}")

    # ── Sauvegarde Parquet ────────────────────────────────────────────────────
    _info(f"Sauvegarde Parquet → {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    features_df.write.mode("overwrite").parquet(str(output_dir))
    _ok(f"Features sauvegardées : {output_dir}")
    _info(f"Durée étape 4 : {time.time() - t0:.1f}s")

    # ── Résumé final ──────────────────────────────────────────────────────────
    elapsed = time.time() - t0_global
    _banner(f"✅ Pipeline données terminé en {elapsed:.0f}s")
    print(f"   CSV source    : {csv_path}")
    print(f"   Features out  : {output_dir}")
    print(f"   Clients       : {n_clients:,}")
    print(f"   Churn rate    : {churned/n_clients*100:.1f}%")
    print()
    print("   Prochaine étape :")
    print(f"     python -m cli.train")
    print(f"     python -m cli.train --features {output_dir}")
    print()

    spark.stop()


if __name__ == "__main__":
    main()
