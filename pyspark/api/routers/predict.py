"""
Router Predict — endpoints de scoring churn.

POST /predict              → score 1 client (requête JSON)
POST /predict/batch        → score jusqu'à 1 000 clients
GET  /predict/top-risk     → top-N clients les plus à risque (depuis Parquet)
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status

from api.dependencies import AppState, get_app_state
from api.schemas.request import BatchPredictRequest, CustomerFeatures
from api.schemas.response import (
    BatchPredictResponse,
    ChurnPrediction,
    SinglePredictResponse,
    TopRiskResponse,
    _risk_level,
)

logger = logging.getLogger("api.routers.predict")
router = APIRouter(prefix="/predict", tags=["Prediction"])


# ──────────────────────────────────────────────────────────────────────────────
# Helper — convertit un dict Spark en ChurnPrediction Pydantic
# ──────────────────────────────────────────────────────────────────────────────

def _to_prediction(row: dict, threshold: float = 0.5) -> ChurnPrediction:
    proba = round(float(row["churn_proba"]), 6)
    return ChurnPrediction(
        customer_id=row["CustomerID"],
        churn_proba=proba,
        prediction=int(row["prediction"]),
        risk_level=_risk_level(proba),
        scored_at=datetime.now(timezone.utc),
    )


def _require_model(state: AppState) -> None:
    """Lève HTTP 503 si le modèle n'est pas encore chargé."""
    if not state.ready or state.model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Modèle non chargé. "
                "Entraînez d'abord le modèle : python -m cli.train"
            ),
        )


# ──────────────────────────────────────────────────────────────────────────────
# POST /predict  — scoring d'un seul client
# ──────────────────────────────────────────────────────────────────────────────

@router.post(
    "",
    response_model=SinglePredictResponse,
    summary="Score un client unique",
    description=(
        "Calcule la probabilité de churn pour un seul client à partir "
        "de ses features RFM. Retourne la proba, le label et le niveau de risque."
    ),
    responses={
        503: {"description": "Modèle non chargé"},
        422: {"description": "Features invalides (validation Pydantic)"},
    },
)
def predict_single(
    customer: CustomerFeatures,
    threshold: float = Query(default=0.5, ge=0.0, le=1.0,
                             description="Seuil de décision churn."),
    state: AppState = Depends(get_app_state),
) -> SinglePredictResponse:
    _require_model(state)

    t0 = time.perf_counter()
    try:
        results = state.score([customer], threshold=threshold)
    except Exception as exc:
        logger.exception("Erreur Spark lors du scoring de %s", customer.customer_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur de scoring : {exc}",
        )

    elapsed_ms = (time.perf_counter() - t0) * 1000
    prediction = _to_prediction(results[0], threshold)

    logger.info(
        "PREDICT  customer=%s  proba=%.4f  risk=%s  time=%.0fms",
        prediction.customer_id, prediction.churn_proba,
        prediction.risk_level, elapsed_ms,
    )
    return SinglePredictResponse(
        prediction=prediction,
        processing_time_ms=round(elapsed_ms, 1),
    )


# ──────────────────────────────────────────────────────────────────────────────
# POST /predict/batch  — scoring d'une liste de clients
# ──────────────────────────────────────────────────────────────────────────────

@router.post(
    "/batch",
    response_model=BatchPredictResponse,
    summary="Score une liste de clients (max 1 000)",
    description=(
        "Calcule le risque de churn pour un batch de clients. "
        "Retourne les prédictions triées par probabilité décroissante, "
        "ainsi que les statistiques agrégées (taux de churn, nb clients à risque)."
    ),
    responses={
        503: {"description": "Modèle non chargé"},
        422: {"description": "Corps de requête invalide"},
    },
)
def predict_batch(
    body: BatchPredictRequest,
    state: AppState = Depends(get_app_state),
) -> BatchPredictResponse:
    _require_model(state)

    t0 = time.perf_counter()
    try:
        results = state.score(body.customers, threshold=body.threshold)
    except Exception as exc:
        logger.exception("Erreur Spark batch scoring (%d clients)", len(body.customers))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur de scoring batch : {exc}",
        )

    elapsed_ms = (time.perf_counter() - t0) * 1000

    predictions = sorted(
        [_to_prediction(r, body.threshold) for r in results],
        key=lambda p: p.churn_proba,
        reverse=True,
    )
    n_churn    = sum(p.prediction for p in predictions)
    n_retained = len(predictions) - n_churn
    churn_rate = round(n_churn / len(predictions), 4) if predictions else 0.0

    logger.info(
        "BATCH  n=%d  churned=%d  rate=%.1f%%  time=%.0fms",
        len(predictions), n_churn, churn_rate * 100, elapsed_ms,
    )
    return BatchPredictResponse(
        predictions=predictions,
        n_scored=len(predictions),
        n_churn=n_churn,
        n_retained=n_retained,
        churn_rate=churn_rate,
        processing_time_ms=round(elapsed_ms, 1),
    )


# ──────────────────────────────────────────────────────────────────────────────
# GET /predict/top-risk  — top-N clients les plus à risque depuis Parquet
# ──────────────────────────────────────────────────────────────────────────────

@router.get(
    "/top-risk",
    response_model=TopRiskResponse,
    summary="Top-N clients les plus à risque de churn",
    description=(
        "Charge un fichier Parquet de features (produit par cli/data_download.py), "
        "score tous les clients, et retourne les N clients avec la plus haute "
        "probabilité de churn. Utile pour cibler les campagnes de rétention."
    ),
    responses={
        404: {"description": "Fichier Parquet des features introuvable"},
        503: {"description": "Modèle non chargé"},
    },
)
def top_risk(
    n: int = Query(default=20, ge=1, le=500,
                   description="Nombre de clients à retourner."),
    features_path: Optional[str] = Query(
        default=None,
        description="Chemin Parquet des features. "
                    "Défaut : data/processed/features (depuis config).",
    ),
    threshold: float = Query(default=0.5, ge=0.0, le=1.0,
                             description="Seuil de décision churn."),
    state: AppState = Depends(get_app_state),
) -> TopRiskResponse:
    _require_model(state)

    # Résoudre le chemin des features
    if features_path is None:
        cfg = state.config
        features_path = str(
            Path(cfg.data.processed_path) / "features"
            if cfg else "data/processed/features"
        )

    if not Path(features_path).exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"Fichier features introuvable : {features_path}. "
                "Lancez d'abord : python -m cli.data_download"
            ),
        )

    t0 = time.perf_counter()
    try:
        # Score tout le dataset et retourne les top-N
        all_results  = state.score_parquet(features_path, threshold=threshold)
        n_total      = len(all_results)
        top_results  = all_results[:n]
    except Exception as exc:
        logger.exception("Erreur Spark top-risk scoring")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur scoring Parquet : {exc}",
        )

    elapsed_ms  = (time.perf_counter() - t0) * 1000
    predictions = [_to_prediction(r, threshold) for r in top_results]

    logger.info(
        "TOP-RISK  n=%d/%d  time=%.0fms",
        len(predictions), n_total, elapsed_ms,
    )
    return TopRiskResponse(
        top_at_risk=predictions,
        n_returned=len(predictions),
        n_total_scored=n_total,
        features_source=features_path,
        processing_time_ms=round(elapsed_ms, 1),
    )
