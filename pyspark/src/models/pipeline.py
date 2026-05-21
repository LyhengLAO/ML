"""
Construction du Pipeline MLlib.

On chaîne :
    VectorAssembler  ->  StandardScaler  ->  Classifier

Le classifieur est paramétré (random_forest par défaut, mais peut
être gbt ou logistic).
"""
from typing import Optional

from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import (
    RandomForestClassifier, GBTClassifier, LogisticRegression
)

from src.config import ModelConfig


def build_pipeline(cfg: Optional[ModelConfig] = None,
                   feature_cols: Optional[list] = None) -> Pipeline:
    """
    Construit le Pipeline MLlib (sans le fitter).

    Parameters
    ----------
    cfg : ModelConfig (algorithme, hyperparamètres).
    feature_cols : liste des colonnes d'entrée à assembler.
    """
    cfg = cfg or ModelConfig()
    feature_cols = feature_cols or [
        "recency", "frequency", "monetary",
        "avg_basket", "n_products", "n_countries", "tenure_days",
    ]

    assembler = VectorAssembler(
        inputCols=feature_cols,
        outputCol="features_raw",
        handleInvalid="skip",
    )
    scaler = StandardScaler(
        inputCol="features_raw",
        outputCol="features",
        withMean=True, withStd=True,
    )
    classifier = _build_classifier(cfg)

    return Pipeline(stages=[assembler, scaler, classifier])


def _build_classifier(cfg: ModelConfig):
    """Instancie le classifieur selon cfg.algorithm."""
    if cfg.algorithm == "random_forest":
        return RandomForestClassifier(
            featuresCol="features",
            labelCol="label",
            numTrees=cfg.num_trees,
            maxDepth=cfg.max_depth,
            minInstancesPerNode=cfg.min_instances_per_node,
            seed=cfg.seed,
        )
    elif cfg.algorithm == "gbt":
        return GBTClassifier(
            featuresCol="features",
            labelCol="label",
            maxIter=cfg.num_trees,
            maxDepth=cfg.max_depth,
            seed=cfg.seed,
        )
    elif cfg.algorithm == "logistic":
        return LogisticRegression(
            featuresCol="features",
            labelCol="label",
            maxIter=100,
        )
    else:
        raise ValueError(f"Algorithme inconnu : {cfg.algorithm}")
