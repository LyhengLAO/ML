"""
Tests unitaires — src/data/schema.py

Vérifie que les schémas Spark sont bien définis (noms, types, nullabilité).
Ces tests ne nécessitent pas de SparkSession.
"""
import pytest
from pyspark.sql.types import (
    DoubleType, IntegerType, StringType, TimestampType,
)

pytestmark = pytest.mark.unit


# ──────────────────────────────────────────────────────────────────────────────
# TRANSACTIONS_SCHEMA
# ──────────────────────────────────────────────────────────────────────────────

class TestTransactionsSchema:

    def test_has_eight_fields(self):
        from src.data.schema import TRANSACTIONS_SCHEMA
        assert len(TRANSACTIONS_SCHEMA.fields) == 8

    def test_column_names_and_order(self):
        from src.data.schema import TRANSACTIONS_SCHEMA
        expected = [
            "InvoiceNo", "StockCode", "Description", "Quantity",
            "InvoiceDate", "UnitPrice", "CustomerID", "Country",
        ]
        assert [f.name for f in TRANSACTIONS_SCHEMA.fields] == expected

    def test_invoice_date_is_timestamp(self):
        from src.data.schema import TRANSACTIONS_SCHEMA
        field = TRANSACTIONS_SCHEMA["InvoiceDate"]
        assert isinstance(field.dataType, TimestampType)

    def test_quantity_is_integer(self):
        from src.data.schema import TRANSACTIONS_SCHEMA
        field = TRANSACTIONS_SCHEMA["Quantity"]
        assert isinstance(field.dataType, IntegerType)

    def test_unit_price_is_double(self):
        from src.data.schema import TRANSACTIONS_SCHEMA
        field = TRANSACTIONS_SCHEMA["UnitPrice"]
        assert isinstance(field.dataType, DoubleType)

    def test_customer_id_is_nullable(self):
        """CustomerID peut être NULL (commandes invités)."""
        from src.data.schema import TRANSACTIONS_SCHEMA
        field = TRANSACTIONS_SCHEMA["CustomerID"]
        assert field.nullable is True

    def test_invoice_no_is_string(self):
        from src.data.schema import TRANSACTIONS_SCHEMA
        field = TRANSACTIONS_SCHEMA["InvoiceNo"]
        assert isinstance(field.dataType, StringType)


# ──────────────────────────────────────────────────────────────────────────────
# FEATURES_SCHEMA
# ──────────────────────────────────────────────────────────────────────────────

class TestFeaturesSchema:

    def test_has_nine_fields(self):
        from src.data.schema import FEATURES_SCHEMA
        assert len(FEATURES_SCHEMA.fields) == 9

    def test_has_label_column(self):
        from src.data.schema import FEATURES_SCHEMA
        names = [f.name for f in FEATURES_SCHEMA.fields]
        assert "label" in names

    def test_has_customer_id_column(self):
        from src.data.schema import FEATURES_SCHEMA
        names = [f.name for f in FEATURES_SCHEMA.fields]
        assert "CustomerID" in names

    def test_rfm_columns_present(self):
        from src.data.schema import FEATURES_SCHEMA
        names = [f.name for f in FEATURES_SCHEMA.fields]
        for col in ("recency", "frequency", "monetary", "avg_basket",
                    "n_products", "n_countries", "tenure_days"):
            assert col in names, f"Colonne RFM manquante : {col}"
