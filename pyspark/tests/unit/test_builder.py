"""
Tests unitaires — src/feature/builder.py

Vérifie la construction des features RFM et du label churn.

Données de référence (clean_df, CUTOFF_DATE = 2021-06-01, HORIZON = 90 j) :

  Passé (< cutoff) :
    INV001  C001  2021-01-15  Revenue = 20   (qty=2, prix=10)
    INV002  C001  2021-03-20  Revenue = 25   (qty=1, prix=25)
    INV003  C002  2021-02-10  Revenue = 30   (qty=3, prix=10)
    INV005  C002  2021-04-01  Revenue = 30   (qty=2, prix=15)

  Futur dans horizon (>= cutoff, < cutoff+90j) :
    INV010  C002  2021-07-05  → C002 actif → label = 0

  Résultat attendu :
    C001 → recency=73, frequency=2, monetary=45.0, label=1  (churné)
    C002 → recency=61, frequency=2, monetary=60.0, label=0  (retenu)
"""
import pytest
from datetime import datetime
from pyspark.sql import functions as F
from tests.conftest import CUTOFF_DATE, HORIZON_DAYS

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def features_rfm(clean_df):
    """Features RFM construites sur les 2 clients de référence."""
    from src.feature.builder import build_features_and_label
    return build_features_and_label(clean_df, CUTOFF_DATE, HORIZON_DAYS)


class TestFeatureColumns:

    def test_all_rfm_columns_present(self, features_rfm):
        """Toutes les colonnes RFM + CustomerID + label doivent être présentes."""
        expected = {
            "CustomerID", "recency", "frequency", "monetary",
            "avg_basket", "n_products", "n_countries", "tenure_days", "label",
        }
        assert set(features_rfm.columns) == expected

    def test_no_extra_columns(self, features_rfm):
        """Aucune colonne parasite (ex. 'active_future') ne doit subsister."""
        assert "active_future" not in features_rfm.columns


class TestClientCount:

    def test_one_row_per_customer(self, features_rfm):
        """
        Chaque client ne doit apparaître qu'une fois.
        (2 clients dans les données de test : C001 et C002)
        """
        total    = features_rfm.count()
        distinct = features_rfm.select("CustomerID").distinct().count()
        assert total == distinct

    def test_exactly_two_customers(self, features_rfm):
        """Uniquement C001 et C002 doivent être présents."""
        customers = {r["CustomerID"] for r in features_rfm.select("CustomerID").collect()}
        assert customers == {"C001", "C002"}


class TestChurnLabel:

    def test_c001_is_churned(self, features_rfm):
        """C001 n'a aucun achat dans la fenêtre future → label = 1."""
        row = features_rfm.filter(F.col("CustomerID") == "C001").collect()[0]
        assert row["label"] == 1

    def test_c002_is_retained(self, features_rfm):
        """C002 a INV010 (2021-07-05) dans la fenêtre future → label = 0."""
        row = features_rfm.filter(F.col("CustomerID") == "C002").collect()[0]
        assert row["label"] == 0


class TestFeatureValues:

    def test_recency_is_positive(self, features_rfm):
        """recency doit être ≥ 0 pour tous les clients."""
        assert features_rfm.filter(F.col("recency") < 0).count() == 0

    def test_frequency_at_least_one(self, features_rfm):
        """Chaque client ayant des achats doit avoir frequency ≥ 1."""
        assert features_rfm.filter(F.col("frequency") < 1).count() == 0

    def test_monetary_positive(self, features_rfm):
        """monetary doit être > 0 (on n'inclut que les achats positifs)."""
        assert features_rfm.filter(F.col("monetary") <= 0).count() == 0

    def test_tenure_days_positive(self, features_rfm):
        """tenure_days = jours entre 1er achat et cutoff, toujours ≥ 0."""
        assert features_rfm.filter(F.col("tenure_days") < 0).count() == 0

    def test_c001_frequency(self, features_rfm):
        """C001 a INV001 + INV002 = 2 factures distinctes dans le passé."""
        row = features_rfm.filter(F.col("CustomerID") == "C001").collect()[0]
        assert row["frequency"] == 2

    def test_c001_monetary(self, features_rfm):
        """C001 : Revenue total = 20 (INV001) + 25 (INV002) = 45."""
        row = features_rfm.filter(F.col("CustomerID") == "C001").collect()[0]
        assert abs(row["monetary"] - 45.0) < 0.01
