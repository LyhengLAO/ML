"""
Schémas Spark des datasets.

Centraliser les schémas dans un seul endroit garantit que toutes les
étapes du pipeline utilisent la même définition (à un seul endroit
à modifier).
"""
from pyspark.sql.types import (
    StructType, StructField,
    StringType, IntegerType, DoubleType, TimestampType,
)


# Schéma du dataset Online Retail II UCI (= notre format canonique)
TRANSACTIONS_SCHEMA = StructType([
    StructField("InvoiceNo",   StringType(),    False),
    StructField("StockCode",   StringType(),    False),
    StructField("Description", StringType(),    True),
    StructField("Quantity",    IntegerType(),   False),
    StructField("InvoiceDate", TimestampType(), False),
    StructField("UnitPrice",   DoubleType(),    False),
    StructField("CustomerID",  StringType(),    True),   # parfois NULL
    StructField("Country",     StringType(),    False),
])


# Schéma des features prêtes pour le ML (sortie de feature engineering)
FEATURES_SCHEMA = StructType([
    StructField("CustomerID",  StringType(),  False),
    StructField("recency",     IntegerType(), True),
    StructField("frequency",   IntegerType(), True),
    StructField("monetary",    DoubleType(),  True),
    StructField("avg_basket",  DoubleType(),  True),
    StructField("n_products",  IntegerType(), True),
    StructField("n_countries", IntegerType(), True),
    StructField("tenure_days", IntegerType(), True),
    StructField("label",       IntegerType(), True),
])
