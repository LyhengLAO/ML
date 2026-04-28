"""
API de scoring - Demo Kubernetes
Avec metriques Prometheus completes
"""
import os
import time
import json
import logging
from datetime import datetime

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, PlainTextResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import psycopg2
from psycopg2.extras import RealDictCursor
import redis
from prometheus_client import (
    Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
)


LOG_LEVEL = os.getenv("LOG_LEVEL", "info").upper()
MODEL_THRESHOLD = float(os.getenv("MODEL_THRESHOLD", "0.5"))
MODEL_VERSION = os.getenv("MODEL_VERSION", "v1.0.0")
DB_HOST = os.getenv("DB_HOST", "postgres")
DB_PORT = int(os.getenv("DB_PORT", "5432"))
DB_NAME = os.getenv("DB_NAME", "mldb")
DB_USER = os.getenv("DB_USER", "mluser")
DB_PASSWORD = os.getenv("DB_PASSWORD", "abcde123")
REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
POD_NAME = os.getenv("POD_NAME", "local")

logging.basicConfig(level=LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] [" + POD_NAME + "] %(message)s")
log = logging.getLogger("ml-api")

app = FastAPI(title="ML Scoring API", version=MODEL_VERSION)
templates = Jinja2Templates(directory="templates")

# ========== Metriques Prometheus ==========
prediction_counter = Counter(
    "ml_predictions_total", "Total predictions",
    ["model_version", "classe", "pod"]
)
prediction_duration = Histogram(
    "ml_prediction_duration_seconds", "Duree prediction",
    ["model_version", "pod"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5)
)
prediction_score_hist = Histogram(
    "ml_prediction_score", "Distribution des scores",
    ["model_version"],
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
)
cache_hits = Counter("ml_cache_hits_total", "Cache hits", ["pod"])
cache_misses = Counter("ml_cache_misses_total", "Cache misses", ["pod"])
db_errors = Counter("ml_db_errors_total", "Erreurs DB", ["pod"])
app_info = Gauge("ml_app_info", "Info app", ["version", "pod"])
app_info.labels(version=MODEL_VERSION, pod=POD_NAME).set(1)

class Client(BaseModel):
    age: int
    revenu: float
    anciennete: int
    dettes: float = 0.0

class Prediction(BaseModel):
    score: float
    classe: int
    pod: str
    model_version: str
    timestamp: str


_pg_conn = None
_redis_conn = None

def get_pg():
    global _pg_conn
    if _pg_conn is None or _pg_conn.closed:
        for attempt in range(5):
            try:
                _pg_conn = psycopg2.connect(
                    host=DB_HOST, port=DB_PORT, dbname=DB_NAME,
                    user=DB_USER, password=DB_PASSWORD, connect_timeout=3
                )
                _pg_conn.autocommit = True
                log.info("Connexion Postgres OK")
                break
            except Exception as e:
                log.warning("Postgres retry %d: %s", attempt + 1, e)
                time.sleep(2)
        else:
            raise RuntimeError("Postgres indisponible")
    return _pg_conn

def get_redis():
    global _redis_conn
    if _redis_conn is None:
        _redis_conn = redis.Redis(host=REDIS_HOST, port=REDIS_PORT,
            decode_responses=True, socket_timeout=3)
    return _redis_conn


def score_model(client: Client) -> float:  # fausse model de scoring pour demo
    base = 0.3
    base += min(client.age / 100, 0.3)
    base += min(client.revenu / 100000, 0.3)
    base += min(client.anciennete / 20, 0.2)
    base -= min(client.dettes / 50000, 0.3)
    x = 0.0
    for i in range(10000):
        x += i * 0.00001
    return max(0.0, min(1.0, base))

@app.get("/health")
def health():
    return {"status": "ok", "pod": POD_NAME}


@app.get("/ready")
def ready():
    try:
        get_pg().cursor().execute("SELECT 1")
        return {"status": "ready", "pod": POD_NAME}
    except Exception as e:
        raise HTTPException(503, f"Not ready: {e}")


@app.get("/info")
def info():
    return {"model_version": MODEL_VERSION, "pod": POD_NAME}

@app.post("/predict", response_model=Prediction)
def predict(client: Client):
    with prediction_duration.labels(model_version=MODEL_VERSION,
                                    pod=POD_NAME).time():
        cache_key = f"score:{client.age}:{int(client.revenu)}:{client.anciennete}:{int(client.dettes)}"
        try:
            r = get_redis()
            cached = r.get(cache_key)
            if cached:
                cache_hits.labels(pod=POD_NAME).inc()
                data = json.loads(cached)
                data["pod"] = POD_NAME
                data["timestamp"] = datetime.utcnow().isoformat()
                prediction_counter.labels(
                    model_version=MODEL_VERSION,
                    classe=str(data["classe"]), pod=POD_NAME).inc()
                return Prediction(**data)
            else:
                cache_misses.labels(pod=POD_NAME).inc()
        except Exception as e:
            log.warning("Redis KO : %s", e)

        score = score_model(client)
        classe = int(score > MODEL_THRESHOLD)
        ts = datetime.utcnow().isoformat()

        prediction_counter.labels(
            model_version=MODEL_VERSION, classe=str(classe), pod=POD_NAME).inc()
        prediction_score_hist.labels(model_version=MODEL_VERSION).observe(score)

        pred = {"score": round(score, 4), "classe": classe,
                "pod": POD_NAME, "model_version": MODEL_VERSION, "timestamp": ts}

        try:
            r = get_redis()
            r.setex(cache_key, 60, json.dumps(pred))
        except Exception:
            pass

        try:
            conn = get_pg()
            with conn.cursor() as c:
                c.execute(
                    "INSERT INTO predictions (age, revenu, anciennete, dettes, score, classe, pod, model_version) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
                    (client.age, client.revenu, client.anciennete,
                     client.dettes, score, classe, POD_NAME, MODEL_VERSION))
        except Exception as e:
            db_errors.labels(pod=POD_NAME).inc()
            log.error("Erreur DB : %s", e)

        return Prediction(**pred)

@app.get("/predictions")
def list_predictions(limit: int = 20):
    try:
        conn = get_pg()
        with conn.cursor(cursor_factory=RealDictCursor) as c:
            c.execute("SELECT * FROM predictions ORDER BY id DESC LIMIT %s", (limit,))
            rows = c.fetchall()
            for row in rows:
                if "created_at" in row and row["created_at"]:
                    row["created_at"] = row["created_at"].isoformat()
            return {"predictions": rows, "count": len(rows)}
    except Exception as e:
        return {"predictions": [], "error": str(e)}


@app.get("/stats")
def stats():
    try:
        conn = get_pg()
        with conn.cursor(cursor_factory=RealDictCursor) as c:
            c.execute(
                "SELECT COUNT(*) as total, AVG(score) as avg_score, "
                "SUM(CASE WHEN classe = 1 THEN 1 ELSE 0 END) as positifs, "
                "COUNT(DISTINCT pod) as nb_pods FROM predictions")
            row = c.fetchone()
            if row and row.get("avg_score"):
                row["avg_score"] = round(float(row["avg_score"]), 4)
            return row or {}
    except Exception as e:
        return {"error": str(e)}


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/metrics")
def metrics():
    """Endpoint Prometheus officiel"""
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)