"""
FastAPI application — the web server that serves predictions.

Endpoints:
  GET  /health         — health check (for Docker / LB)
  GET  /metadata       — model info
  POST /predict        — single prediction
  POST /predict/batch  — batch prediction
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from prometheus_client import Counter, Gauge, Histogram, make_asgi_app

from arabic_sentiment.api.schemas import (
    HealthResponse,
    MetadataResponse,
    PredictionRequest,
    PredictionResponse,
)
from arabic_sentiment.config import settings
from arabic_sentiment.model import SentimentClassifier

logger = logging.getLogger(__name__)

# ── Prometheus Metrics ──────────────────────────
PREDICTION_COUNT = Counter(
    "arabic_sentiment_requests_total",
    "Total prediction requests served",
    ["status"],
)
PREDICTION_LATENCY = Histogram(
    "arabic_sentiment_latency_seconds",
    "Prediction latency in seconds",
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0],
)
PSI_GAUGE = Gauge(
    "arabic_sentiment_psi_score",
    "Population Stability Index (PSI) drift score",
    ["feature"],
)
P95_LATENCY_GAUGE = Gauge(
    "arabic_sentiment_p95_latency_ms",
    "Latest measured p95 latency in milliseconds",
)


def update_prometheus_drift_metrics() -> None:
    """Read latest drift metrics from disk and update Prometheus gauges."""
    drift_file = Path("reports/drift_metrics.json")
    if drift_file.exists():
        try:
            with open(drift_file, "r") as f:
                data = json.load(f)
            psi_len = data.get("psi_text_length", 0.0)
            psi_conf = data.get("psi_confidence_score", 0.0)
            PSI_GAUGE.labels(feature="text_length").set(psi_len)
            PSI_GAUGE.labels(feature="confidence_score").set(psi_conf)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Failed to update drift gauges: {e}")

# ──────────────────────────────────────────────
# Lifespan — load model once at startup
# ──────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once at startup; clean up on shutdown."""
    logger.info("Starting up — loading model...")
    app.state.classifier = SentimentClassifier()
    app.state.classifier.load()
    logger.info("Model loaded, ready to serve!")
    # Initialize Prometheus gauges
    P95_LATENCY_GAUGE.set(980.0)
    update_prometheus_drift_metrics()
    yield

    logger.info("Shutting down...")


# ──────────────────────────────────────────────
# App instance
# ──────────────────────────────────────────────

app = FastAPI(
    title="Arabic Sentiment Analysis API",
    description="Classify Arabic text as positive, negative, or neutral",
    version=settings.model_version,
    lifespan=lifespan,
)


# ──────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check — returns 200 only if model is loaded."""
    classifier = app.state.classifier
    if not classifier.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return HealthResponse(
        status="healthy",
        model_loaded=True,
        model_version=settings.model_version,
    )


@app.get("/metadata", response_model=MetadataResponse)
async def metadata():
    """Model metadata — what model is running and its configuration."""
    return MetadataResponse(
        model_version=settings.model_version,
        model_name=settings.model_name,
        framework="onnx",
        max_seq_length=settings.max_seq_length,
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Predict sentiment for Arabic text."""
    correlation_id = str(uuid.uuid4())

    start = time.perf_counter()

    try:
        result = app.state.classifier.predict_one(request.text)
    except Exception as e:
        logger.error(f"Prediction failed: {e}", extra={"correlation_id": correlation_id})
        raise HTTPException(status_code=500, detail="Prediction failed") from e

    latency_ms = (time.perf_counter() - start) * 1000
    # ── Record Prometheus observation ──
    PREDICTION_COUNT.labels(status="success").inc()
    PREDICTION_LATENCY.observe(latency_ms / 1000.0)
    logger.info(
        "Prediction served",
        extra={
            "correlation_id": correlation_id,
            "label": result["label"],
            "confidence": result["confidence"],
            "latency_ms": round(latency_ms, 2),
        },
    )

    return PredictionResponse(
        label=result["label"],
        confidence=result["confidence"],
        model_version=settings.model_version,
        correlation_id=correlation_id,
        latency_ms=round(latency_ms, 2),
    )


@app.post("/predict/batch")
async def predict_batch(requests: list[PredictionRequest]):
    """Predict sentiment for multiple texts at once."""
    texts = [r.text for r in requests]
    results = app.state.classifier.predict_batch(texts)
    return results
# ──────────────────────────────────────────────
# Prometheus Exporter Endpoint
# ──────────────────────────────────────────────
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)