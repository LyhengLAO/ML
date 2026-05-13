"""
Nettoyage des transactions.

Étapes typiques sur des données e-commerce :
1. Supprimer les lignes sans CustomerID (clients invités, inutiles
   pour un modèle de churn par client).
2. Filtrer les retours / annulations (Quantity <= 0 ou prix <= 0).
3. Calculer le revenu par ligne = Quantity * UnitPrice.
4. Dédupliquer.
"""
from pyspark.sql import DataFrame, functions as F


def clean_transactions(df: DataFrame) -> DataFrame:
    """
    Applique le nettoyage standard sur un DataFrame de transactions.

    Returns
    -------
    DataFrame nettoyé avec une colonne `Revenue` ajoutée.
    """
    df_clean = (
        df
        # 1. Pas de client => on ne peut pas faire du churn par client
        .filter(F.col("CustomerID").isNotNull())
        # 2. Retours et erreurs
        .filter(F.col("Quantity") > 0)
        .filter(F.col("UnitPrice") > 0)
        # 3. Revenu par ligne
        .withColumn("Revenue", F.col("Quantity") * F.col("UnitPrice"))
        # 4. Doublons exacts
        .dropDuplicates()
    )
    return df_clean


def quality_report(df: DataFrame) -> dict:
    """
    Calcule un mini-rapport de qualité sur le DataFrame.
    Utile à logger avant/après le nettoyage.
    """
    total = df.count()
    nulls_per_col = (
        df.select([
            F.sum(F.col(c).isNull().cast("int")).alias(c)
            for c in df.columns
        ]).collect()[0].asDict()
    )
    return {
        "total_rows": total,
        "nulls_per_column": nulls_per_col,
        "distinct_customers": (
            df.select("CustomerID").distinct().count()
            if "CustomerID" in df.columns else None
        ),
    }
