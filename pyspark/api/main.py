"""
Application FastAPI — Churn Prediction API.

Point d'entrée principal du serveur. Configure :
  - Le lifespan (démarrage Spark + chargement modèle / arrêt propre)
  - Les routers (health, predict)
  - Le middleware (CORS, logs de requête, gestion d'erreurs)
  - La documentation OpenAPI automatique

Lancer avec :
    python -m cli.serve
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
"""
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.dependencies import app_state
from api.routers import health as health_router
from api.routers import predict as predict_router

# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("api.main")


# ──────────────────────────────────────────────────────────────────────────────
# Lifespan — startup & shutdown
# ──────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(application: FastAPI):
    """
    Cycle de vie FastAPI :
      startup  → initialise SparkSession + charge le modèle
      shutdown → arrête la SparkSession proprement
    """
    logger.info("=" * 60)
    logger.info("  Churn Prediction API — démarrage")
    logger.info("=" * 60)

    # Démarrage (bloquant mais exécuté dans le thread principal)
    # On utilise un thread pour ne pas bloquer la boucle asyncio
    import asyncio
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, app_state.initialize)

    logger.info("API prête  ✅  (model_loaded=%s)", app_state.ready)
    yield

    # Arrêt
    logger.info("Arrêt de l'API...")
    await loop.run_in_executor(None, app_state.shutdown)
    logger.info("API arrêtée proprement.")


# ──────────────────────────────────────────────────────────────────────────────
# Application FastAPI
# ──────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Churn Prediction API",
    description=(
        "## API de scoring de churn client\n\n"
        "Basée sur un modèle **PySpark ML** (Random Forest / GBT / Logistic Regression) "
        "entraîné sur des features **RFM** (Recency, Frequency, Monetary).\n\n"
        "### Workflow\n"
        "1. `python -m cli.data_download` — prépare les features\n"
        "2. `python -m cli.train`          — entraîne le modèle\n"
        "3. `python -m cli.serve`          — démarre cette API\n\n"
        "### Endpoints principaux\n"
        "- `POST /predict`        — score 1 client\n"
        "- `POST /predict/batch`  — score jusqu'à 1 000 clients\n"
        "- `GET  /predict/top-risk` — top-N clients à risque\n"
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)


# ──────────────────────────────────────────────────────────────────────────────
# Middleware
# ──────────────────────────────────────────────────────────────────────────────

# CORS — à restreindre en production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],         # restreindre en prod : ["https://monsite.com"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log chaque requête avec méthode, path et temps de réponse."""
    t0 = time.perf_counter()
    response = await call_next(request)
    elapsed = (time.perf_counter() - t0) * 1000
    logger.info(
        "%s %s → %d  (%.0f ms)",
        request.method, request.url.path,
        response.status_code, elapsed,
    )
    return response


# ──────────────────────────────────────────────────────────────────────────────
# Gestion globale des erreurs
# ──────────────────────────────────────────────────────────────────────────────

@app.exception_handler(RuntimeError)
async def runtime_error_handler(request: Request, exc: RuntimeError):
    return JSONResponse(
        status_code=503,
        content={"detail": str(exc), "error_type": "RuntimeError"},
    )


@app.exception_handler(FileNotFoundError)
async def file_not_found_handler(request: Request, exc: FileNotFoundError):
    return JSONResponse(
        status_code=404,
        content={"detail": str(exc), "error_type": "FileNotFoundError"},
    )


# ──────────────────────────────────────────────────────────────────────────────
# Routers
# ──────────────────────────────────────────────────────────────────────────────

app.include_router(health_router.router)
app.include_router(predict_router.router)


# ──────────────────────────────────────────────────────────────────────────────
# Route racine
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
def root():
    return {
        "name":    "Churn Prediction API",
        "version": "1.0.0",
        "docs":    "/docs",
        "health":  "/health",
        "status":  "/status",
        "endpoints": {
            "score_one":   "POST /predict",
            "score_batch": "POST /predict/batch",
            "top_risk":    "GET  /predict/top-risk?n=20",
        },
    }
