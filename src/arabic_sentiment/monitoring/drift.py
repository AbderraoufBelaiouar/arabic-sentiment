"""
Data drift and statistical stability monitoring.

Capabilities:
1. Computes Population Stability Index (PSI) for text_length and confidence_score.
2. Generates an interactive Evidently AI DataDrift report in reports/data_drift_report.html.
3. Saves machine-readable drift metrics to reports/drift_metrics.json.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

try:
    from evidently.metric_preset import DataDriftPreset
    from evidently.report import Report
except ImportError:
    from evidently.legacy.metric_preset import DataDriftPreset
    from evidently.legacy.report import Report

from arabic_sentiment.config import PROJECT_ROOT, REPORTS_DIR

DRIFT_REPORT_HTML = REPORTS_DIR / "data_drift_report.html"
DRIFT_METRICS_JSON = REPORTS_DIR / "drift_metrics.json"
REFERENCE_CSV = REPORTS_DIR / "drift_reference.csv"

def calculate_psi(expected: np.ndarray, actual: np.ndarray, num_buckets: int = 10) -> float:
    """
    Calculate Population Stability Index (PSI) between two numeric distributions.
    
    Standard industry benchmarks:
    - PSI < 0.1: No significant drift (Stable).
    - 0.1 <= PSI <= 0.25: Moderate drift (Watchlist).
    - PSI > 0.25: Significant drift (Triggers alert and model retraining).
    """
    if len(expected) == 0 or len(actual) == 0:
        return 0.0

    percentiles = np.linspace(0, 100, num_buckets + 1)
    bins = np.percentile(expected, percentiles)
    bins[0] = -np.inf
    bins[-1] = np.inf
    bins = np.unique(bins)
    if len(bins) < 2:
        return 0.0

    expected_counts, _ = np.histogram(expected, bins=bins)
    actual_counts, _ = np.histogram(actual, bins=bins)

    eps = 1e-4
    expected_pct = (expected_counts / len(expected)) + eps
    actual_pct = (actual_counts / len(actual)) + eps

    expected_pct /= np.sum(expected_pct)
    actual_pct /= np.sum(actual_pct)

    psi_val = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(round(psi_val, 4))


def generate_drift_analysis(current_df: pd.DataFrame) -> dict[str, Any]:
    """
    Compare current scored batch against the training baseline.
    Generates both Evidently HTML report and PSI metrics.
    """
    print("🔍 [Monitoring] Running data drift analysis...")

    # 1. Build Reference DataFrame from training set
    if not REFERENCE_CSV.exists():
        raise FileNotFoundError(
            f"No drift reference found at {REFERENCE_CSV}. Generate it with:\n"
            "  uv run python -m arabic_sentiment.monitoring.make_reference"
        )
    ref_df = pd.read_csv(REFERENCE_CSV)[["text_length", "confidence_score"]]

    curr_features = current_df[["text_length", "confidence_score"]].copy()

    # 2. Compute PSI for text_length and confidence_score
    psi_length = calculate_psi(ref_df["text_length"].to_numpy(), curr_features["text_length"].to_numpy())
    psi_confidence = calculate_psi(ref_df["confidence_score"].to_numpy(), curr_features["confidence_score"].to_numpy())

    # 3. Generate Evidently HTML DataDrift report
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref_df, current_data=curr_features)
    
    DRIFT_REPORT_HTML.parent.mkdir(parents=True, exist_ok=True)
    report.save_html(str(DRIFT_REPORT_HTML))
    print(f"📊 Evidently DataDrift HTML report saved to: {DRIFT_REPORT_HTML}")

    # 4. Save drift metrics JSON
    drift_status = "ALERT_DRIFT" if (psi_length > 0.25 or psi_confidence > 0.25) else "STABLE"
    metrics_summary = {
        "psi_text_length": psi_length,
        "psi_confidence_score": psi_confidence,
        "psi_threshold": 0.25,
        "status": drift_status,
        "total_current_samples": len(current_df),
        "mean_current_confidence": float(round(curr_features["confidence_score"].mean(), 4)),
        "reference_samples": len(ref_df),
    }

    with open(DRIFT_METRICS_JSON, "w") as f:
        json.dump(metrics_summary, f, indent=2)
    print(f"💾 Drift metrics saved to: {DRIFT_METRICS_JSON}")

    print(f"📈 PSI text_length       : {psi_length} (Threshold: 0.25)")
    print(f"📈 PSI confidence_score  : {psi_confidence} (Threshold: 0.25)")
    if drift_status == "ALERT_DRIFT":
        print("⚠️ [ALERT] Distribution drift detected! PSI exceeds 0.25.")
    else:
        print("✅ [OK] Distributions within stable boundaries.")

    return metrics_summary


if __name__ == "__main__":
    latest_csv = PROJECT_ROOT / "data" / "scoring" / "output" / "predictions_latest.csv"
    if latest_csv.exists():
        df = pd.read_csv(latest_csv)
        generate_drift_analysis(df)
    else:
        print("❌ No batch scored data found. Run src/arabic_sentiment/batch_score.py first.")
