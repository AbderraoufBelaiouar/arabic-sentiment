"""Evaluate the production ONNX artifact on the held-out test split.

Generates reports/eval_metrics.json, consumed by the CI quality gate.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from sklearn.metrics import accuracy_score, f1_score

from arabic_sentiment.config import REPORTS_DIR, settings
from arabic_sentiment.data import load_arabic_sentiment_dataset
from arabic_sentiment.model import LABEL_MAP, SentimentClassifier

EVAL_METRICS_JSON = REPORTS_DIR / "eval_metrics.json"
NAME_TO_IDX = {name: idx for idx, name in LABEL_MAP.items()}


def artifact_fingerprint(model_dir: Path) -> str:
    """SHA-256 over all files of the exported model directory."""
    digest = hashlib.sha256()
    for path in sorted(model_dir.rglob("*")):
        if path.is_file():
            digest.update(path.name.encode())
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    """Run end-to-end evaluation and persist metrics."""
    ds = load_arabic_sentiment_dataset()["test"]
    texts = list(ds["text"])
    labels = [int(label) for label in ds["label"]]

    classifier = SentimentClassifier()
    classifier.load()

    print(f"🔍 Evaluating {len(texts)} samples ...")
    results = classifier.predict_batch(texts)
    predictions = [NAME_TO_IDX[result["label"]] for result in results]

    metrics = {
        "accuracy": round(float(accuracy_score(labels, predictions)), 4),
        "f1_macro": round(float(f1_score(labels, predictions, average="macro")), 4),
        "n_samples": len(texts),
        "model_path": str(settings.model_path),
        "model_sha256": artifact_fingerprint(Path(settings.model_path)),
        "evaluated_at": datetime.now(UTC).isoformat(),
    }

    EVAL_METRICS_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(EVAL_METRICS_JSON, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"📊 accuracy: {metrics['accuracy']}  |  f1_macro: {metrics['f1_macro']}")
    print(f"💾 Saved → {EVAL_METRICS_JSON}")


if __name__ == "__main__":
    main()
