"""
Schémas Pydantic — corps des requêtes entrantes.
"""
from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field, model_validator


# ──────────────────────────────────────────────────────────────────────────────
# Features d'un client (entrée du modèle)
# ──────────────────────────────────────────────────────────────────────────────

class CustomerFeatures(BaseModel):
    """
    Features RFM d'un client, calculées AVANT la cutoff_date.
    Toutes les valeurs doivent être strictement positives (ou ≥ 0 pour recency).
    """
    customer_id: str = Field(
        ...,
        description="Identifiant unique du client.",
        examples=["C12345"],
    )
    recency: int = Field(
        ..., ge=0,
        description="Jours écoulés depuis le dernier achat (0 = achat très récent).",
        examples=[30],
    )
    frequency: int = Field(
        ..., ge=1,
        description="Nombre de factures distinctes.",
        examples=[8],
    )
    monetary: float = Field(
        ..., gt=0,
        description="Revenu total cumulé en euros.",
        examples=[450.75],
    )
    avg_basket: float = Field(
        ..., gt=0,
        description="Valeur moyenne du panier (monetary / frequency).",
        examples=[56.34],
    )
    n_products: int = Field(
        ..., ge=1,
        description="Nombre de références produits (SKU) distinctes achetées.",
        examples=[12],
    )
    n_countries: int = Field(
        ..., ge=1,
        description="Nombre de pays distincts dans les commandes.",
        examples=[1],
    )
    tenure_days: int = Field(
        ..., ge=0,
        description="Ancienneté client en jours (1er achat → cutoff).",
        examples=[365],
    )

    model_config = {"json_schema_extra": {
        "example": {
            "customer_id": "C12345",
            "recency": 30,
            "frequency": 8,
            "monetary": 450.75,
            "avg_basket": 56.34,
            "n_products": 12,
            "n_countries": 1,
            "tenure_days": 365,
        }
    }}


# ──────────────────────────────────────────────────────────────────────────────
# Requête batch
# ──────────────────────────────────────────────────────────────────────────────

class BatchPredictRequest(BaseModel):
    """Corps de la requête POST /predict/batch."""
    customers: List[CustomerFeatures] = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Liste de 1 à 1 000 clients à scorer.",
    )
    threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Seuil de décision : churn_proba ≥ threshold → prediction=1.",
    )

    @model_validator(mode="after")
    def check_unique_customer_ids(self) -> "BatchPredictRequest":
        ids = [c.customer_id for c in self.customers]
        if len(ids) != len(set(ids)):
            duplicates = [cid for cid in set(ids) if ids.count(cid) > 1]
            raise ValueError(
                f"customer_id dupliqués dans la requête : {duplicates[:5]}"
            )
        return self
