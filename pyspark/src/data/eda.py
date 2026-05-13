"""
Analyse exploratoire (EDA).

Calcule et affiche un résumé statistique du dataset. Sur Spark, on
agrège distribué puis on `show()` ou on convertit en pandas pour
visualiser localement les petits résultats.
"""
from pyspark.sql import DataFrame, functions as F


def eda_summary(df: DataFrame) -> None:
    """Affiche un résumé exploratoire dans la console."""
    print("=" * 70)
    print(" EDA SUMMARY")
    print("=" * 70)

    print("\n--- Schéma ---")
    df.printSchema()

    print(f"\n--- Volumes ---")
    n_rows = df.count()
    n_cust = df.select("CustomerID").distinct().count()
    n_inv  = df.select("InvoiceNo").distinct().count()
    n_sku  = df.select("StockCode").distinct().count()
    print(f"Lignes        : {n_rows:>12,}")
    print(f"Clients       : {n_cust:>12,}")
    print(f"Factures      : {n_inv:>12,}")
    print(f"Produits      : {n_sku:>12,}")

    print(f"\n--- Période couverte ---")
    df.agg(
        F.min("InvoiceDate").alias("date_min"),
        F.max("InvoiceDate").alias("date_max"),
    ).show(truncate=False)

    print(f"\n--- Top 5 pays par revenu ---")
    (df.groupBy("Country")
       .agg(F.round(F.sum("Revenue"), 2).alias("revenu_total"),
            F.countDistinct("CustomerID").alias("nb_clients"))
       .orderBy(F.col("revenu_total").desc())
       .show(5, truncate=False))

    print(f"\n--- Top 5 produits par quantité vendue ---")
    (df.groupBy("StockCode", "Description")
       .agg(F.sum("Quantity").alias("qte_totale"),
            F.round(F.sum("Revenue"), 2).alias("revenu_total"))
       .orderBy(F.col("qte_totale").desc())
       .show(5, truncate=False))

    print(f"\n--- Stats du panier (Quantity, UnitPrice, Revenue) ---")
    df.select("Quantity", "UnitPrice", "Revenue").summary(
        "count", "mean", "stddev", "min", "25%", "50%", "75%", "max"
    ).show()
