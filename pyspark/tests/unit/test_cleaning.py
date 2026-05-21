"""
Tests unitaires — src/data/cleaning.py

Chaque test cible une règle de nettoyage précise en utilisant
le fixture `raw_transactions` (9 lignes avec cas limites connus).

Lignes brutes attendues : 9
Lignes valides après nettoyage : 5  (INV001, INV002, INV003, INV005, INV010)
"""
import pytest
from pyspark.sql import functions as F

pytestmark = pytest.mark.unit


class TestCleanTransactions:

    def test_null_customer_id_removed(self, clean_df):
        """Toute ligne sans CustomerID doit être éliminée (INV004)."""
        assert clean_df.filter(F.col("CustomerID").isNull()).count() == 0

    def test_negative_quantity_removed(self, clean_df):
        """Les retours (Quantity ≤ 0) doivent être supprimés (INV006 : qty=-1)."""
        assert clean_df.filter(F.col("Quantity") <= 0).count() == 0

    def test_zero_price_removed(self, clean_df):
        """Les lignes avec prix nul ou négatif doivent être supprimées (INV007)."""
        assert clean_df.filter(F.col("UnitPrice") <= 0).count() == 0

    def test_exact_duplicates_removed(self, raw_transactions, clean_df):
        """
        INV001 apparaît deux fois dans raw_transactions.
        Après nettoyage, il ne doit rester qu'une seule occurrence.
        """
        n_raw   = raw_transactions.filter(F.col("InvoiceNo") == "INV001").count()
        n_clean = clean_df.filter(F.col("InvoiceNo") == "INV001").count()
        assert n_raw == 2
        assert n_clean == 1

    def test_revenue_column_added(self, clean_df):
        """clean_transactions doit ajouter la colonne Revenue."""
        assert "Revenue" in clean_df.columns

    def test_revenue_equals_qty_times_price(self, clean_df):
        """Revenue = Quantity * UnitPrice pour chaque ligne."""
        wrong = (
            clean_df
            .withColumn("expected", F.col("Quantity") * F.col("UnitPrice"))
            .filter(F.abs(F.col("Revenue") - F.col("expected")) > 0.001)
            .count()
        )
        assert wrong == 0

    def test_final_row_count(self, raw_transactions, clean_df):
        """
        9 lignes brutes − 4 invalides (null, retour, prix nul, doublon) = 5.
        """
        assert raw_transactions.count() == 9
        assert clean_df.count() == 5

    def test_all_remaining_customers_known(self, clean_df):
        """Seuls C001 et C002 doivent survivre au nettoyage."""
        customers = {r["CustomerID"] for r in clean_df.select("CustomerID").distinct().collect()}
        assert customers == {"C001", "C002"}


class TestQualityReport:

    def test_report_has_required_keys(self, raw_transactions):
        """quality_report doit retourner un dict avec les 3 clés attendues."""
        from src.data.cleaning import quality_report
        report = quality_report(raw_transactions)
        assert "total_rows"        in report
        assert "nulls_per_column"  in report
        assert "distinct_customers" in report

    def test_total_rows_correct(self, raw_transactions):
        """total_rows doit correspondre au count réel du DataFrame."""
        from src.data.cleaning import quality_report
        report = quality_report(raw_transactions)
        assert report["total_rows"] == 9

    def test_nulls_per_column_is_dict(self, raw_transactions):
        """nulls_per_column doit être un dict avec une entrée par colonne."""
        from src.data.cleaning import quality_report
        report = quality_report(raw_transactions)
        assert isinstance(report["nulls_per_column"], dict)

    def test_customer_id_null_count(self, raw_transactions):
        """Il y a exactement 1 CustomerID NULL dans raw_transactions (INV004)."""
        from src.data.cleaning import quality_report
        report = quality_report(raw_transactions)
        assert report["nulls_per_column"].get("CustomerID", 0) == 1
