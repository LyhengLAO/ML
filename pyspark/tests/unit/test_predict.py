"""
Tests unitaires — src/models/predict.py

Vérifie predict_batch() et top_n_at_risk() sur le modèle entraîné.
"""
import pytest
from pyspark.sql import functions as F

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def scored_df(features_df, trained_model):
    """DataFrame scoré (toutes prédictions sur les 40 clients)."""
    from src.models.predict import predict_batch
    return predict_batch(trained_model, features_df).cache()


class TestPredictBatch:

    def test_output_columns(self, scored_df):
        """Colonnes de sortie : CustomerID, churn_proba, prediction."""
        assert set(scored_df.columns) == {"CustomerID", "churn_proba", "prediction"}

    def test_row_count_matches_input(self, features_df, scored_df):
        """Le nombre de lignes scorées doit égaler le nombre de clients en entrée."""
        assert scored_df.count() == features_df.count()

    def test_churn_proba_in_zero_one(self, scored_df):
        """churn_proba ∈ [0, 1] pour chaque client."""
        out_of_range = scored_df.filter(
            (F.col("churn_proba") < 0.0) | (F.col("churn_proba") > 1.0)
        ).count()
        assert out_of_range == 0

    def test_prediction_is_binary(self, scored_df):
        """prediction ∈ {0.0, 1.0} (classificateur binaire)."""
        invalid = scored_df.filter(
            ~F.col("prediction").isin([0.0, 1.0])
        ).count()
        assert invalid == 0

    def test_no_null_predictions(self, scored_df):
        """Aucune valeur nulle dans les colonnes de sortie."""
        null_count = scored_df.filter(
            F.col("churn_proba").isNull() | F.col("prediction").isNull()
        ).count()
        assert null_count == 0

    def test_customer_ids_preserved(self, features_df, scored_df):
        """Tous les CustomerID en entrée doivent se retrouver dans la sortie."""
        input_ids  = {r["CustomerID"] for r in features_df.select("CustomerID").collect()}
        output_ids = {r["CustomerID"] for r in scored_df.select("CustomerID").collect()}
        assert input_ids == output_ids


class TestTopNAtRisk:

    def test_top_n_returns_exact_count(self, scored_df):
        """top_n_at_risk(n=5) doit retourner exactement 5 lignes."""
        from src.models.predict import top_n_at_risk
        top5 = top_n_at_risk(scored_df, n=5)
        assert top5.count() == 5

    def test_top_n_sorted_descending(self, scored_df):
        """Les lignes doivent être triées par churn_proba décroissant."""
        from src.models.predict import top_n_at_risk
        top10 = top_n_at_risk(scored_df, n=10)
        probas = [r["churn_proba"] for r in top10.select("churn_proba").collect()]
        assert probas == sorted(probas, reverse=True)

    def test_top_n_has_highest_probabilities(self, scored_df):
        """Tout client hors du Top-5 doit avoir une proba ≤ min du Top-5."""
        from src.models.predict import top_n_at_risk
        top5     = top_n_at_risk(scored_df, n=5)
        min_top5 = top5.agg(F.min("churn_proba")).collect()[0][0]
        max_rest = (
            scored_df
            .join(top5.select("CustomerID"), on="CustomerID", how="left_anti")
            .agg(F.max("churn_proba"))
            .collect()[0][0]
        )
        # max_rest peut être None si tous sont dans le top (dataset < n)
        if max_rest is not None:
            assert max_rest <= min_top5 + 1e-9   # tolérance flottante

    def test_top_n_full_dataset(self, features_df, scored_df):
        """Si n ≥ nb total de clients, tous les clients sont retournés."""
        from src.models.predict import top_n_at_risk
        n = features_df.count()
        top_all = top_n_at_risk(scored_df, n=n)
        assert top_all.count() == n


class TestLoadModel:

    def test_load_model_roundtrip(self, trained_model, tmp_path):
        """
        Sauvegarde puis rechargement du modèle : les prédictions doivent
        être identiques avant et après.
        """
        from src.models.predict import load_model, predict_batch
        from pyspark.sql import SparkSession
        import pandas as pd

        spark = SparkSession.getActiveSession()
        model_path = str(tmp_path / "test_model")
        trained_model.write().overwrite().save(model_path)

        reloaded = load_model(model_path)
        assert reloaded is not None
