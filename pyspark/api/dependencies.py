"""
Gestion du cycle de vie de l'application :
  - SparkSession unique (singleton)
  - PipelineModel chargé au démarrage
  - Injection de dépendances FastAPI

Variables d'environnement reconnues :
  APP_ENV     : local | production  (défaut: local)
  MODEL_DIR   : chemin du PipelineModel  (défaut: output/model)
"""
from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger("api.dependencies")


# ──────────────────────────────────────────────────────────────────────────────
# Schéma Spark pour les requêtes d'inférence (sans label)
# ──────────────────────────────────────────────────────────────────────────────

def _inference_schema():
    from pyspark.sql.types import (
        DoubleType, IntegerType, StringType, StructField, StructType,
    )
    return StructType([
        StructField("CustomerID",  StringType(),  False),
        StructField("recency",     IntegerType(), False),
        StructField("frequency",   IntegerType(), False),
        StructField("monetary",    DoubleType(),  False),
        StructField("avg_basket",  DoubleType(),  False),
        StructField("n_products",  IntegerType(), False),
        StructField("n_countries", IntegerType(), False),
        StructField("tenure_days", IntegerType(), False),
    ])


# ──────────────────────────────────────────────────────────────────────────────
# AppState — singleton global partagé par tous les workers
# ──────────────────────────────────────────────────────────────────────────────

class AppState:
    """
    Conteneur d'état applicatif.

    Initialisé une seule fois au démarrage de FastAPI (lifespan),
    partagé (thread-safe) par tous les handlers de route.
    """

    def __init__(self) -> None:
        self.spark      = None           # SparkSession
        self.model      = None           # PipelineModel
        self.config     = None           # src.config.Config
        self.model_path: str = ""
        self.ready: bool = False
        self.started_at: Optional[datetime] = None
        self._schema    = None           # cache du schéma Spark d'inférence

    # ── Initialisation (appelé par le lifespan FastAPI) ───────────────────────
    def initialize(self) -> None:
        """Démarre Spark et charge le modèle. Bloquant — appelé au startup."""
        env        = os.getenv("APP_ENV", "local")
        model_dir  = os.getenv("MODEL_DIR", "output/model")

        logger.info("Initialisation de l'API — env=%s, model=%s", env, model_dir)
        t0 = time.perf_counter()

        # 1. Config
        from src.config import load_config
        self.config = load_config(env)
        self.model_path = model_dir

        # 2. SparkSession
        from src.spark.session import get_spark
        self.spark = get_spark(self.config.spark, app_name="ChurnAPI")
        logger.info("SparkSession démarrée (%s)", self.config.spark.master)

        # 3. Modèle
        model_path = Path(model_dir)
        if not model_path.exists():
            logger.warning(
                "Modèle introuvable : %s — API démarrée en mode dégradé. "
                "Entraînez d'abord avec : python -m cli.train",
                model_dir,
            )
        else:
            from src.models.predict import load_model
            self.model = load_model(model_dir)
            logger.info("Modèle chargé : %s", model_dir)

        # 4. Cache schéma inférence
        self._schema = _inference_schema()

        self.ready      = self.model is not None
        self.started_at = datetime.now(timezone.utc)
        elapsed = (time.perf_counter() - t0) * 1000
        logger.info("API prête en %.0f ms  (model_loaded=%s)", elapsed, self.ready)

    # ── Arrêt (appelé par le lifespan FastAPI) ────────────────────────────────
    def shutdown(self) -> None:
        """Arrête la SparkSession proprement."""
        if self.spark:
            logger.info("Arrêt de la SparkSession...")
            self.spark.stop()
            self.spark = None
        self.ready = False
        logger.info("API arrêtée.")

    # ── Uptime ────────────────────────────────────────────────────────────────
    @property
    def uptime_seconds(self) -> float:
        if self.started_at is None:
            return 0.0
        return (datetime.now(timezone.utc) - self.started_at).total_seconds()

    # ── Scoring ───────────────────────────────────────────────────────────────
    def score(self, customers: list, threshold: float = 0.5) -> list:
        """
        Score une liste de CustomerFeatures (Pydantic) via le PipelineModel.

        Retourne une liste de dicts :
          { customer_id, churn_proba, prediction }

        Raises
        ------
        RuntimeError si le modèle n'est pas chargé.
        """
        if not self.ready or self.model is None:
            raise RuntimeError(
                "Modèle non chargé. Lancez d'abord : python -m cli.train"
            )

        from pyspark.sql import Row
        from pyspark.sql import functions as F

        # Construire les lignes Spark depuis les objets Pydantic
        rows = [
            Row(
                CustomerID=c.customer_id,
                recency=c.recency,
                frequency=c.frequency,
                monetary=float(c.monetary),
                avg_basket=float(c.avg_basket),
                n_products=c.n_products,
                n_countries=c.n_countries,
                tenure_days=c.tenure_days,
            )
            for c in customers
        ]

        df = self.spark.createDataFrame(rows, schema=self._schema)

        proba_udf = F.udf(lambda v: float(v[1]), "double")
        scored = (
            self.model.transform(df)
            .withColumn("churn_proba", proba_udf("probability"))
            .withColumn(
                "prediction",
                F.when(F.col("churn_proba") >= threshold, 1).otherwise(0),
            )
            .select("CustomerID", "churn_proba", "prediction")
        )
        return [row.asDict() for row in scored.collect()]

    def score_parquet(self, parquet_path: str, threshold: float = 0.5,
                      top_n: Optional[int] = None) -> list:
        """
        Score un fichier Parquet de features et retourne les résultats.
        Utilisé par GET /predict/top-risk.
        """
        if not self.ready or self.model is None:
            raise RuntimeError("Modèle non chargé.")

        from pyspark.sql import functions as F

        features_df = self.spark.read.parquet(parquet_path)
        proba_udf   = F.udf(lambda v: float(v[1]), "double")

        scored = (
            self.model.transform(features_df)
            .withColumn("churn_proba", proba_udf("probability"))
            .withColumn(
                "prediction",
                F.when(F.col("churn_proba") >= threshold, 1).otherwise(0),
            )
            .select("CustomerID", "churn_proba", "prediction")
            .orderBy(F.col("churn_proba").desc())
        )
        if top_n is not None:
            scored = scored.limit(top_n)

        return [row.asDict() for row in scored.collect()]


# ──────────────────────────────────────────────────────────────────────────────
# Instance globale — injectée dans les routes via FastAPI Depends
# ──────────────────────────────────────────────────────────────────────────────

app_state = AppState()


def get_app_state() -> AppState:
    """Dependency FastAPI : fournit l'AppState aux handlers de route."""
    return app_state
