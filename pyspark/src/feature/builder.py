"""
Construction des features RFM avec horizon temporel.

Logique métier :
1. La cutoff_date sépare le passé (calcul des features) du futur
   (observation du label).
2. Les features sont calculées UNIQUEMENT sur les transactions
   antérieures à la cutoff (pas de fuite de données).
3. Le label `churned` vaut 1 si le client n'a aucun achat dans la
   fenêtre [cutoff, cutoff + horizon_days].
"""
from datetime import datetime, timedelta
from typing import Optional

from pyspark.sql import DataFrame, functions as F

from src.config import FeaturesConfig


def build_features_and_label(df_clean: DataFrame,
                             cutoff_date: datetime,
                             horizon_days: int = 90) -> DataFrame:
    """
    Construit le dataset d'apprentissage : 1 ligne par client.

    Features produites :
    - recency       : jours entre dernier achat et cutoff
    - frequency     : nombre de factures distinctes
    - monetary      : revenu total
    - avg_basket    : panier moyen
    - n_products    : nb de SKU distincts
    - n_countries   : nb de pays distincts
    - tenure_days   : jours entre 1er achat et cutoff
    - label         : 1 si pas d'achat dans les `horizon_days` jours
                      après cutoff, sinon 0.
    """
    cutoff = F.lit(cutoff_date)
    horizon_end = cutoff_date + timedelta(days=horizon_days)

    past = df_clean.filter(F.col("InvoiceDate") < cutoff)
    future = df_clean.filter(
        (F.col("InvoiceDate") >= cutoff) &
        (F.col("InvoiceDate") < F.lit(horizon_end))
    )

    features = (
        past.groupBy("CustomerID")
            .agg(
                F.datediff(cutoff, F.max("InvoiceDate")).alias("recency"),
                F.countDistinct("InvoiceNo").alias("frequency"),
                F.round(F.sum("Revenue"), 2).alias("monetary"),
                F.round(F.avg("Revenue"), 2).alias("avg_basket"),
                F.countDistinct("StockCode").alias("n_products"),
                F.countDistinct("Country").alias("n_countries"),
                F.datediff(cutoff, F.min("InvoiceDate")).alias("tenure_days"),
            )
    )

    active_in_future = (
        future.select("CustomerID")
              .distinct()
              .withColumn("active_future", F.lit(1))
    )

    return (
        features.join(active_in_future, "CustomerID", "left")
                .withColumn(
                    "label",
                    F.when(F.col("active_future").isNull(), 1).otherwise(0)
                )
                .drop("active_future")
    )


def build_from_config(df_clean: DataFrame,
                      cfg: Optional[FeaturesConfig] = None) -> DataFrame:
    """Wrapper qui prend un FeaturesConfig au lieu de paramètres bruts."""
    cfg = cfg or FeaturesConfig()
    cutoff = datetime.strptime(cfg.cutoff_date, "%Y-%m-%d")
    return build_features_and_label(df_clean, cutoff, cfg.horizon_days)
