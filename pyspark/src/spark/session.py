"""
Factory de SparkSession.

Centralise la création de la SparkSession pour que la configuration
soit cohérente partout dans le projet.
"""
from typing import Optional

from pyspark.sql import SparkSession

from src.config import SparkConfig


def get_spark(cfg: Optional[SparkConfig] = None,
              app_name: Optional[str] = None) -> SparkSession:
    """
    Crée (ou récupère) une SparkSession configurée.

    Parameters
    ----------
    cfg : SparkConfig (depuis Config). Si None, valeurs par défaut.
    app_name : permet de surcharger le nom de l'app.

    Returns
    -------
    SparkSession prête à l'emploi.
    """
    cfg = cfg or SparkConfig()
    spark = (
        SparkSession.builder
        .appName(app_name or cfg.app_name)
        .master(cfg.master)
        .config("spark.sql.shuffle.partitions", str(cfg.shuffle_partitions))
        .config("spark.sql.adaptive.enabled", str(cfg.adaptive_enabled).lower())
        .config("spark.sql.adaptive.coalescePartitions.enabled",
                str(cfg.adaptive_enabled).lower())
        .config("spark.driver.memory", cfg.driver_memory)
        .config("spark.executor.memory", cfg.executor_memory)
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark
