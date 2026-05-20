"""
Inférence avec un modèle entraîné.
"""
from pyspark.ml import PipelineModel
from pyspark.sql import DataFrame, SparkSession, functions as F


def load_model(path: str) -> PipelineModel:
    """Charge un PipelineModel sauvegardé."""
    return PipelineModel.load(path)


def predict_batch(model: PipelineModel, features_df: DataFrame) -> DataFrame:
    """
    Score un DataFrame de features.
    Retourne CustomerID, prediction (0/1), churn_proba (float).
    """
    proba_udf = F.udf(lambda v: float(v[1]), "double")
    return (model.transform(features_df)
                 .withColumn("churn_proba", proba_udf("probability"))
                 .select("CustomerID", "churn_proba", "prediction"))


def top_n_at_risk(scored_df: DataFrame, n: int = 20) -> DataFrame:
    """Renvoie les N clients les plus à risque de churn."""
    return scored_df.orderBy(F.col("churn_proba").desc()).limit(n)
