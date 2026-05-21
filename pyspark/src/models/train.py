"""
Entraînement du modèle.

Fournit `train()` qui prend un train_df et retourne un PipelineModel,
et `split_data()` pour diviser train/test.
"""
from typing import Optional, Tuple

from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql import DataFrame

from src.config import ModelConfig
from src.models.pipeline import build_pipeline


def split_data(df: DataFrame,
               train_ratio: float = 0.8,
               seed: int = 42) -> Tuple[DataFrame, DataFrame]:
    """Split aléatoire train/test."""
    return df.randomSplit([train_ratio, 1 - train_ratio], seed=seed)


def train(train_df: DataFrame,
          cfg: Optional[ModelConfig] = None,
          feature_cols: Optional[list] = None) -> PipelineModel:
    """
    Entraîne le pipeline ML, avec ou sans cross-validation.

    Parameters
    ----------
    train_df : DataFrame d'entraînement.
    cfg : ModelConfig (incluant use_cv).
    feature_cols : colonnes d'entrée.

    Returns
    -------
    PipelineModel entraîné.
    """
    cfg = cfg or ModelConfig()
    pipeline = build_pipeline(cfg, feature_cols)

    if not cfg.use_cv:
        return pipeline.fit(train_df)

    return _train_with_cv(pipeline, train_df, cfg)


def _train_with_cv(pipeline: Pipeline,
                   train_df: DataFrame,
                   cfg: ModelConfig) -> PipelineModel:
    """Cross-validation à 3 plis avec une petite grille."""
    rf = pipeline.getStages()[-1]
    grid = (ParamGridBuilder()
            .addGrid(rf.numTrees, [50, 100])
            .addGrid(rf.maxDepth, [6, 10])
            .build())

    evaluator = BinaryClassificationEvaluator(
        labelCol="label",
        rawPredictionCol="rawPrediction",
        metricName="areaUnderROC",
    )
    cv = CrossValidator(
        estimator=pipeline,
        estimatorParamMaps=grid,
        evaluator=evaluator,
        numFolds=3,
        seed=cfg.seed,
        parallelism=2,
    )
    return cv.fit(train_df).bestModel
