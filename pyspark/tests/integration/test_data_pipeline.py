"""
Tests d'intégration — Pipeline de données (cli/data_download.py)

Vérifie que les étapes du pipeline de données s'enchaînent correctement :
  download.load_csv → cleaning.clean_transactions → builder.build_features_and_label → Parquet

Ces tests utilisent :
  - Des fichiers CSV temporaires (tmp_path) — pas de téléchargement réseau
  - Le fixture spark de conftest
  - La fixture clean_df comme point de départ intermédiaire

Marqueur : @pytest.mark.integration
Lance séparément avec : pytest -m integration
"""
import pytest
import pandas as pd
from datetime import datetime
from pathlib import Path
from pyspark.sql import functions as F

pytestmark = pytest.mark.integration

# Cutoff pour les tests d'intégration
CUTOFF = datetime(2021, 6, 1)
HORIZON = 90


# ──────────────────────────────────────────────────────────────────────────────
# Tests pipeline données — étapes enchaînées
# ──────────────────────────────────────────────────────────────────────────────

class TestCleanThenFeatures:
    """Teste l'enchaînement clean_transactions → build_features_and_label."""

    def test_pipeline_produces_one_row_per_customer(self, clean_df):
        """Le résultat doit avoir 1 ligne par client unique."""
        from src.feature.builder import build_features_and_label
        features = build_features_and_label(clean_df, CUTOFF, HORIZON)
        total    = features.count()
        distinct = features.select("CustomerID").distinct().count()
        assert total == distinct

    def test_pipeline_label_column_present(self, clean_df):
        """La colonne 'label' doit exister dans le résultat final."""
        from src.feature.builder import build_features_and_label
        features = build_features_and_label(clean_df, CUTOFF, HORIZON)
        assert "label" in features.columns

    def test_pipeline_no_null_labels(self, clean_df):
        """Chaque client doit avoir un label (0 ou 1), jamais NULL."""
        from src.feature.builder import build_features_and_label
        features = build_features_and_label(clean_df, CUTOFF, HORIZON)
        null_labels = features.filter(F.col("label").isNull()).count()
        assert null_labels == 0

    def test_pipeline_label_is_binary(self, clean_df):
        """label ∈ {0, 1} — pas d'autres valeurs."""
        from src.feature.builder import build_features_and_label
        features = build_features_and_label(clean_df, CUTOFF, HORIZON)
        invalid = features.filter(~F.col("label").isin([0, 1])).count()
        assert invalid == 0


class TestCSVLoadPipeline:
    """Teste le chargement d'un CSV local → nettoyage → features."""

    @pytest.fixture
    def local_csv(self, tmp_path) -> Path:
        """Crée un petit CSV de transactions dans un répertoire temporaire."""
        csv_path = tmp_path / "transactions.csv"
        pd.DataFrame({
            "InvoiceNo":   ["I001", "I002", "I003", "I004", "I001"],  # I001 dupliqué
            "StockCode":   ["SKU1", "SKU2", "SKU1", "SKU3", "SKU1"],
            "Description": ["Alpha", "Beta", "Alpha", "Gamma", "Alpha"],
            "Quantity":    [2, 1, 3, -1, 2],              # I004 : retour
            "InvoiceDate": [
                "2021-01-15 10:00:00",
                "2021-03-20 14:00:00",
                "2021-07-05 16:00:00",   # futur (après cutoff)
                "2021-02-01 10:00:00",
                "2021-01-15 10:00:00",   # doublon exact de I001
            ],
            "UnitPrice":   [10.0, 25.0, 10.0, 10.0, 10.0],
            "CustomerID":  ["C001", "C001", "C002", "C001", "C001"],
            "Country":     ["UK", "UK", "FR", "UK", "UK"],
        }).to_csv(csv_path, index=False)
        return csv_path

    def test_csv_loads_into_spark(self, spark, local_csv):
        """load_csv doit lire le CSV avec le bon schéma."""
        from src.data.download import load_csv
        df = load_csv(spark, str(local_csv))
        assert df.count() == 5          # 5 lignes dans le CSV
        assert "InvoiceDate" in df.columns

    def test_clean_removes_invalid_rows(self, spark, local_csv):
        """Après nettoyage : retour et doublon supprimés."""
        from src.data.download import load_csv
        from src.data.cleaning import clean_transactions
        raw   = load_csv(spark, str(local_csv))
        clean = clean_transactions(raw)
        # I004 (retour, qty=-1) supprimé + doublon I001 supprimé → 3 lignes
        assert clean.count() == 3

    def test_full_csv_to_features_pipeline(self, spark, local_csv):
        """CSV → clean → features doit produire des features valides."""
        from src.data.download import load_csv
        from src.data.cleaning import clean_transactions
        from src.feature.builder import build_features_and_label

        raw      = load_csv(spark, str(local_csv))
        clean    = clean_transactions(raw)
        features = build_features_and_label(clean, CUTOFF, HORIZON)

        # C001 et C002 (si assez de données dans le passé)
        assert features.count() >= 1
        assert "label" in features.columns
        assert "recency" in features.columns


class TestParquetRoundtrip:
    """Teste que les features peuvent être sauvées et rechargées via Parquet."""

    def test_save_and_reload_preserves_schema(self, spark, clean_df, tmp_path):
        """Le schéma doit être identique après un aller-retour Parquet."""
        from src.feature.builder import build_features_and_label
        features     = build_features_and_label(clean_df, CUTOFF, HORIZON)
        parquet_path = str(tmp_path / "features.parquet")

        features.write.mode("overwrite").parquet(parquet_path)
        reloaded = spark.read.parquet(parquet_path)

        assert set(reloaded.columns) == set(features.columns)

    def test_save_and_reload_preserves_row_count(self, spark, clean_df, tmp_path):
        """Le nombre de lignes doit être identique après sauvegarde Parquet."""
        from src.feature.builder import build_features_and_label
        features     = build_features_and_label(clean_df, CUTOFF, HORIZON)
        parquet_path = str(tmp_path / "features_count.parquet")

        features.write.mode("overwrite").parquet(parquet_path)
        reloaded = spark.read.parquet(parquet_path)

        assert reloaded.count() == features.count()

    def test_reloaded_labels_unchanged(self, spark, clean_df, tmp_path):
        """Les labels doivent être identiques après reload Parquet."""
        from src.feature.builder import build_features_and_label
        features     = build_features_and_label(clean_df, CUTOFF, HORIZON)
        parquet_path = str(tmp_path / "features_label.parquet")

        features.write.mode("overwrite").parquet(parquet_path)
        reloaded = spark.read.parquet(parquet_path)

        orig_labels    = {r["CustomerID"]: r["label"] for r in features.collect()}
        reload_labels  = {r["CustomerID"]: r["label"] for r in reloaded.collect()}
        assert orig_labels == reload_labels
