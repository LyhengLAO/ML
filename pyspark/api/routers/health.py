"""
Router Health — endpoints de surveillance.

GET /health   → vérification rapide (liveness probe)
GET /status   → état détaillé (readiness probe + infos modèle)
"""
from fastapi import APIRouter, Depends
from api.dependencies import AppState, get_app_state
from api.schemas.response import HealthResponse, StatusResponse

router = APIRouter(tags=["Health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness probe",
    description="Vérifie que le serveur répond et indique l'état de Spark / modèle.",
)
def health(state: AppState = Depends(get_app_state)) -> HealthResponse:
    spark_active = state.spark is not None
    model_loaded = state.model is not None

    if model_loaded and spark_active:
        status = "ok"
    elif spark_active:
        status = "degraded"   # Spark OK mais modèle absent
    else:
        status = "unavailable"

    return HealthResponse(
        status=status,
        model_loaded=model_loaded,
        spark_active=spark_active,
        uptime_seconds=round(state.uptime_seconds, 1),
    )


@router.get(
    "/status",
    response_model=StatusResponse,
    summary="Readiness probe + informations détaillées",
    description=(
        "Retourne la configuration complète de l'API : "
        "modèle chargé, SparkSession, feature cols, uptime."
    ),
)
def status(state: AppState = Depends(get_app_state)) -> StatusResponse:
    cfg = state.config

    spark_master   = cfg.spark.master   if cfg else "N/A"
    spark_app_name = cfg.spark.app_name if cfg else "N/A"
    feature_cols   = cfg.features.feature_cols if cfg else []
    env            = cfg.env if cfg else "unknown"

    spark_active = state.spark is not None
    model_loaded = state.model is not None

    return StatusResponse(
        status="ok" if model_loaded else "degraded",
        environment=env,
        model_path=state.model_path,
        model_loaded=model_loaded,
        spark_master=spark_master,
        spark_app_name=spark_app_name,
        feature_cols=feature_cols,
        uptime_seconds=round(state.uptime_seconds, 1),
        started_at=state.started_at,
    )
