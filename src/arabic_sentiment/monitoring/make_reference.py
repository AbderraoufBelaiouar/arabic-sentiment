"""Build the drift reference artifact from the served model.

Samples the validation split, scores it with the SAME ONNX artifact that
serves traffic, and stores text_length + confidence_score distributions.
The DVC stage `drift-reference` regenerates it on every model update.
"""

from __future__ import annotations
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from arabic_sentiment.config import REPORTS_DIR, settings
from arabic_sentiment.data import load_arabic_sentiment_dataset
from arabic_sentiment.evaluate import artifact_fingerprint
from arabic_sentiment.model import SentimentClassifier
from arabic_sentiment.monitoring.drift import REFERENCE_CSV

REFERENCE_META = REPORTS_DIR / "drift_reference_meta.json"
SAMPLE_SIZE = 1000
SEED = 42


def main() -> None:
    ds = load_arabic_sentiment_dataset()["validation"]
    texts = list(ds["text"])

    rng = np.random.default_rng(SEED)
    n = min(SAMPLE_SIZE, len(texts))
    indices = rng.choice(len(texts), size=n, replace=False)
    sample = [texts[i] for i in indices]

    classifier = SentimentClassifier()
    classifier.load()

    print(f"🔍 Scoring {n} validation samples with the served ONNX model ...")
    results = classifier.predict_batch(sample)

    ref_df = pd.DataFrame(
        {
            "text_length": [len(t) for t in sample],
            "confidence_score": [r["confidence"] for r in results],
        }
    )

    REFERENCE_CSV.parent.mkdir(parents=True, exist_ok=True)
    ref_df.to_csv(REFERENCE_CSV, index=False)

    meta = {
        "model_path": str(settings.model_path),
        "model_sha256": artifact_fingerprint(Path(settings.model_path)),
        "n_samples": n,
        "seed": SEED,
        "created_at": datetime.now(UTC).isoformat(),
    }
    with open(REFERENCE_META, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"💾 Reference saved → {REFERENCE_CSV} ({n} samples)")
    print(
        f"   confidence range: "
        f"{ref_df['confidence_score'].min():.4f} – {ref_df['confidence_score'].max():.4f}"
    )


if __name__ == "__main__":
    main()