"""
Fixtures partagées pour toute la suite de tests.

Hiérarchie des fixtures (toutes session-scoped pour éviter de redémarrer Spark) :

  spark                 SparkSession locale légère (1 worker, UI off)
  ├── raw_transactions  DataFrame brut 9 lignes (valides + cas limites)
  │   └── clean_df      Après clean_transactions() → 5 lignes propres
  └── features_df       40 clients synthétiques (20 churnés + 20 retenus)
          └── trained_model   PipelineModel entraîné (RF 5 arbres)

Données synthétiques (raw_transactions) :
  INV001 C001 passé valide   ─┐ C001 → churn  (label=1)
  INV002 C001 passé valide   ─┘
  INV003 C002 passé valide   ─┐ C002 → retenu (label=0)
  INV005 C002 passé valide    │ (a un achat futur INV010)
  INV010 C002 futur valide   ─┘
  INV004 null  → supprimé (CustomerID NULL)
  INV006 C001  → supprimé (Quantity = -1)
  INV007 C001  → supprimé (UnitPrice = 0)
  INV001 C001  → supprimé (doublon exact d'INV001)

Cutoff : 2021-06-01  |  Horizon : 90 jours (→ 2021-08-30)
"""
import random
from datetime import datetime

import pytest
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    DoubleType, IntegerType, StringType,
    StructField, StructType, TimestampType,
)

# ──────────────────────────────────────────────────────────────────────────────
# Constantes partagées entre les tests
# ──────────────────────────────────────────────────────────────────────────────
CUTOFF_DATE  = datetime(2021, 6, 1)
HORIZON_DAYS = 90   # fenêtre future : jusqu'au 2021-08-30

FEATURE_COLS = [
    "recency", "frequency", "monetary",
    "avg_basket", "n_products", "n_countries", "tenure_days",
]

# Schéma des features utilisé pour créer les DataFrames de test ML
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


# ──────────────────────────────────────────────────────────────────────────────
# SparkSession — partagée pour toute la session de tests
# ──────────────────────────────────────────────────────────────────────────────
@pytest.fixture(scope="session")
def spark():
    """SparkSession légère (1 thread, UI désactivée, logs ERROR seulement)."""
    session = (
        SparkSession.builder
        .master("local[1]")
        .appName("ChurnPipeline-Tests")
        .config("spark.sql.shuffle.partitions", "2")
        .config("spark.driver.memory", "1g")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.adaptive.enabled", "false")   # résultats déterministes
        .getOrCreate()
    )
    session.sparkContext.setLogLevel("ERROR")
    yield session
    session.stop()


# ──────────────────────────────────────────────────────────────────────────────
# Transactions brutes (9 lignes avec tous les cas limites)
# ──────────────────────────────────────────────────────────────────────────────
@pytest.fixture(scope="session")
def raw_transactions(spark):
    """
    DataFrame de transactions synthétiques couvrant tous les cas de nettoyage.
    Utilise le TRANSACTIONS_SCHEMA canonique du projet.
    """
    from src.data.schema import TRANSACTIONS_SCHEMA

    data = [
        # ── Valides — passé (avant cutoff 2021-06-01) ──────────────────────
        ("INV001", "SKU_A", "Widget Alpha",  2,  datetime(2021, 1, 15, 10, 0),  10.0, "C001", "UK"),
        ("INV002", "SKU_B", "Widget Beta",   1,  datetime(2021, 3, 20, 14, 0),  25.0, "C001", "UK"),
        ("INV003", "SKU_A", "Widget Alpha",  3,  datetime(2021, 2, 10,  9, 0),  10.0, "C002", "FR"),
        ("INV005", "SKU_C", "Widget Gamma",  2,  datetime(2021, 4,  1, 11, 0),  15.0, "C002", "FR"),
        # ── Valide — futur dans horizon (→ C002 retenu, label=0) ───────────
        ("INV010", "SKU_B", "Widget Beta",   1,  datetime(2021, 7,  5, 16, 0),  25.0, "C002", "FR"),
        # ── Invalides — doivent être supprimés par clean_transactions ───────
        ("INV004", "SKU_A", "Widget Alpha",  1,  datetime(2021, 4,  1,  8, 0),   5.0,  None,  "DE"),  # pas de client
        ("INV006", "SKU_A", "Widget Alpha", -1,  datetime(2021, 2,  1, 10, 0),  10.0, "C001", "UK"),  # retour
        ("INV007", "SKU_D", "Widget Delta",  2,  datetime(2021, 3,  1, 12, 0),   0.0, "C001", "UK"),  # prix nul
        ("INV001", "SKU_A", "Widget Alpha",  2,  datetime(2021, 1, 15, 10, 0),  10.0, "C001", "UK"),  # doublon exact
    ]
    return spark.createDataFrame(data, schema=TRANSACTIONS_SCHEMA)


# ──────────────────────────────────────────────────────────────────────────────
# DataFrame nettoyé (5 lignes valides)
# ──────────────────────────────────────────────────────────────────────────────
@pytest.fixture(scope="session")
def clean_df(spark, raw_transactions):
    """
    Transactions après clean_transactions() — 5 lignes propres :
    INV001, INV002, INV003, INV005 (passé) + INV010 (futur).
    Colonnes : TRANSACTIONS_SCHEMA + Revenue.
    """
    from src.data.cleaning import clean_transactions
    return clean_transactions(raw_transactions)


# ──────────────────────────────────────────────────────────────────────────────
# Features ML — 40 clients synthétiques (pour les tests de modèle)
# ──────────────────────────────────────────────────────────────────────────────
@pytest.fixture(scope="session")
def features_df(spark):
    """
    40 clients synthétiques avec profils RFM bien séparés :
      - 20 churnés  (recency élevée, fréquence faible, monetary faible)
      - 20 retenus  (recency faible,  fréquence forte,  monetary forte)

    Le signal est volontairement fort pour que le RF atteigne une bonne AUC
    même avec seulement 5 arbres.
    """
    rng = random.Random(42)
    rows = []

    # 20 clients churnés
    for i in range(20):
        rows.append((
            f"CHURN_{i:03d}",
            rng.randint(60, 200),                    # recency élevée
            rng.randint(1, 3),                       # fréquence faible
            round(rng.uniform(20.0, 150.0), 2),      # monetary faible
            round(rng.uniform(10.0, 50.0), 2),       # avg_basket
            rng.randint(1, 3),                       # peu de produits
            1,                                       # 1 pays
            rng.randint(90, 400),                    # tenure_days
            1,                                       # label = churné
        ))

    # 20 clients retenus
    for i in range(20):
        rows.append((
            f"RETAIN_{i:03d}",
            rng.randint(1, 30),                      # recency faible
            rng.randint(8, 20),                      # fréquence forte
            round(rng.uniform(300.0, 1000.0), 2),    # monetary forte
            round(rng.uniform(30.0, 100.0), 2),      # avg_basket
            rng.randint(5, 15),                      # beaucoup de produits
            rng.randint(1, 3),                       # n_countries
            rng.randint(200, 700),                   # tenure_days long
            0,                                       # label = retenu
        ))

    return spark.createDataFrame(rows, schema=FEATURES_SCHEMA)


# ──────────────────────────────────────────────────────────────────────────────
# Modèle entraîné (partagé pour evaluate + predict)
# ──────────────────────────────────────────────────────────────────────────────
@pytest.fixture(scope="session")
def trained_model(features_df):
    """
    PipelineModel entraîné sur features_df avec un petit Random Forest
    (5 arbres, profondeur 3) — assez pour tester, assez rapide pour les tests.
    """
    from src.config import ModelConfig
    from src.models.train import split_data, train

    cfg = ModelConfig(
        algorithm="random_forest",
        num_trees=5,
        max_depth=3,
        use_cv=False,
        seed=42,
    )
    train_df, _ = split_data(features_df, train_ratio=0.8, seed=42)
    return train(train_df, cfg, FEATURE_COLS)
