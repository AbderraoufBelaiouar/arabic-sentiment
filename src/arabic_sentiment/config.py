"""
Configuration management using pydantic-settings.

Settings come from environment variables with sensible defaults.
In development: uses defaults.
In Docker:      override via env vars (e.g. MODEL_PATH=/app/models/arabert-twitter-onnx).
"""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings

# ──────────────────────────────────────────────
# Derived paths (used by train.py, not by Settings)
# ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # arabic-sentiment/
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
DATA_DIR = PROJECT_ROOT / "data" / "raw"

# ──────────────────────────────────────────────
# Dataset constants (training only)
# ──────────────────────────────────────────────
DATASET_NAME = "mteb/tweet_sentiment_multilingual"
DATASET_SUBSET = "arabic"
LABEL_NAMES = ["negative", "neutral", "positive"]
NUM_LABELS = len(LABEL_NAMES)

# ──────────────────────────────────────────────
# Training hyperparameters
# ──────────────────────────────────────────────
NUM_EPOCHS = 5
TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 64
LEARNING_RATE = 2e-5
LR_SCHEDULER = "cosine"
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 100
EARLY_STOPPING_PATIENCE = 2
METRIC_FOR_BEST_MODEL = "f1_macro"


class Settings(BaseSettings):
    """Application settings — all configurable values live here."""

    model_path: str = str(MODELS_DIR / "arabert-twitter-onnx")
    model_name: str = "aubmindlab/bert-base-arabertv02-twitter"
    api_port: int = 8000
    model_version: str = "0.1.0"
    max_seq_length: int = 128
    log_level: str = "INFO"


# Singleton instance used across the app
settings = Settings()
