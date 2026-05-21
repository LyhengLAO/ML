"""
Schémas Pydantic — corps des réponses sortantes.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, computed_field


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _risk_level(proba: float) -> str:
    """Convertit une probabilité de churn en niveau de risque lisible."""
    if proba < 0.33:
        return "LOW"
    if proba < 0.66:
        return "MEDIUM"
    return "HIGH"


# ──────────────────────────────────────────────────────────────────────────────
# Prédiction individuelle
# ──────────────────────────────────────────────────────────────────────────────

class ChurnPrediction(BaseModel):
    """Score de churn pour un client."""
    customer_id: str        = Field(..., description="Identifiant du client.")
    churn_proba: float      = Field(..., ge=0.0, le=1.0, description="Probabilité de churn [0–1].")
    prediction: int         = Field(..., description="Label prédit : 0=retenu, 1=churné.")
    risk_level: str         = Field(..., description="Niveau de risque : LOW | MEDIUM | HIGH.")
    scored_at: datetime     = Field(..., description="Horodatage de la prédiction (UTC).")

    model_config = {"json_schema_extra": {
        "example": {
            "customer_id": "C12345",
            "churn_proba": 0.82,
            "prediction": 1,
            "risk_level": "HIGH",
            "scored_at": "2026-05-21T10:00:00Z",
        }
    }}


# ──────────────────────────────────────────────────────────────────────────────
# Réponse d'un seul client (POST /predict)
# ──────────────────────────────────────────────────────────────────────────────

class SinglePredictResponse(BaseModel):
    prediction: ChurnPrediction
    processing_time_ms: float = Field(..., description="Temps de traitement Spark (ms).")


# ──────────────────────────────────────────────────────────────────────────────
# Réponse batch (POST /predict/batch)
# ──────────────────────────────────────────────────────────────────────────────

class BatchPredictResponse(BaseModel):
    predictions: List[ChurnPrediction]
    n_scored: int           = Field(..., description="Nombre de clients scorés.")
    n_churn: int            = Field(..., description="Clients prédits comme churnés.")
    n_retained: int         = Field(..., description="Clients prédits comme retenus.")
    churn_rate: float       = Field(..., description="Taux de churn prédit (n_churn / n_scored).")
    processing_time_ms: float = Field(..., description="Temps de traitement Spark (ms).")


# ──────────────────────────────────────────────────────────────────────────────
# Réponse top-N clients à risque (GET /predict/top-risk)
# ──────────────────────────────────────────────────────────────────────────────

class TopRiskResponse(BaseModel):
    top_at_risk: List[ChurnPrediction]
    n_returned: int
    n_total_scored: int
    features_source: str    = Field(..., description="Chemin Parquet des features utilisées.")
    processing_time_ms: float


# ──────────────────────────────────────────────────────────────────────────────
# Health & Status
# ──────────────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str             = Field(..., description="ok | degraded | unavailable")
    model_loaded: bool
    spark_active: bool
    uptime_seconds: float


class StatusResponse(BaseModel):
    status: str
    version: str            = Field(default="1.0.0")
    environment: str
    model_path: str
    model_loaded: bool
    spark_master: str
    spark_app_name: str
    feature_cols: List[str]
    uptime_seconds: float
    started_at: datetime


# ──────────────────────────────────────────────────────────────────────────────
# Erreurs
# ──────────────────────────────────────────────────────────────────────────────

class ErrorResponse(BaseModel):
    detail: str
    error_type: Optional[str] = None
