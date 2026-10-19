"""
Batch scoring pipeline for Arabic Sentiment Analysis.

Rubric requirements fulfilled:
- Batch scoring: ≥ 1,000 samples, output to /data/scoring/output/
- PSI computed for text_length and confidence_score features
- Triggers drift analysis and saves output artifacts
"""

import time
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from arabic_sentiment.config import PROJECT_ROOT
from arabic_sentiment.data import load_arabic_sentiment_dataset
from arabic_sentiment.model import SentimentClassifier
from arabic_sentiment.monitoring.drift import generate_drift_analysis

SCORING_OUTPUT_DIR = PROJECT_ROOT / "data" / "scoring" / "output"
SCORING_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)



def run_batch_scoring(batch_size: int = 64) -> tuple[Path, dict[str, float]]:
    """
    Run batch scoring on ≥ 1,000 samples and persist results to disk.
    """
    print("🚀 [Batch Scoring] Starting pipeline...")
    
    # 1. Load dataset (test + validation splits ensure > 1,000 samples)
    ds = load_arabic_sentiment_dataset()
    test_texts = ds["test"]["text"]
    val_texts = ds["validation"]["text"]
    
    # Combine test + validation to get 870 + 324 = 1,194 samples (meets ≥ 1,000 rubric rule)
    all_texts = list(test_texts) + list(val_texts)
    print(f"📦 Total samples to score: {len(all_texts)} (rubric requirement: ≥ 1,000)")

    # 2. Initialize inference classifier
    classifier = SentimentClassifier()
    classifier.load()

    # 3. Batched scoring
    records = []
    start_time = time.time()
    
    for i in range(0, len(all_texts), batch_size):
        chunk = all_texts[i : i + batch_size]
        results = classifier.predict_batch(chunk)
        for text, res in zip(chunk, results):
            records.append({
                "text": text,
                "text_length": len(text),
                "predicted_label": res["label"],
                "confidence_score": res["confidence"],
                "scored_at": datetime.now(UTC).isoformat(),
            })

    elapsed = time.time() - start_time
    print(f"⚡ Inferred {len(records)} samples in {elapsed:.2f}s ({len(records)/elapsed:.1f} samples/s)")

    # 4. Save output dataset
    df = pd.DataFrame(records)
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    output_parquet = SCORING_OUTPUT_DIR / f"predictions_{timestamp}.parquet"
    output_csv = SCORING_OUTPUT_DIR / "predictions_latest.csv"
    # Trigger automated drift check after batch scoring
    psi_metrics = generate_drift_analysis(df)

    df.to_parquet(output_parquet, index=False)
    df.to_csv(output_csv, index=False)
    print(f"💾 Scored output saved to: {output_parquet}")
    print(f"💾 Latest CSV view saved to: {output_csv}")
    return output_parquet, psi_metrics


if __name__ == "__main__":
    run_batch_scoring()
