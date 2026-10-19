"""Tests for drift monitoring: PSI maths and reference-artifact behaviour."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from arabic_sentiment.monitoring import drift


def test_psi_identical_distributions_is_zero():
    values = np.linspace(0.5, 1.0, 200)
    assert drift.calculate_psi(values, values) == 0.0


def test_psi_shifted_distribution_is_large():
    rng = np.random.default_rng(0)
    expected = rng.uniform(0.5, 1.0, 1000)
    actual = expected + 0.5
    assert drift.calculate_psi(expected, actual) > 0.25


def test_psi_empty_returns_zero():
    assert drift.calculate_psi(np.array([]), np.array([1.0])) == 0.0


def test_missing_reference_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(drift, "REFERENCE_CSV", tmp_path / "missing.csv")
    current = pd.DataFrame({"text_length": [10], "confidence_score": [0.9]})
    with pytest.raises(FileNotFoundError):
        drift.generate_drift_analysis(current)


def test_stable_run_writes_metrics(tmp_path, monkeypatch):
    rng = np.random.default_rng(1)
    reference = pd.DataFrame(
        {
            "text_length": rng.integers(10, 120, 500),
            "confidence_score": rng.uniform(0.5, 1.0, 500),
        }
    )
    ref_path = tmp_path / "ref.csv"
    reference.to_csv(ref_path, index=False)

    monkeypatch.setattr(drift, "REFERENCE_CSV", ref_path)
    monkeypatch.setattr(drift, "DRIFT_METRICS_JSON", tmp_path / "drift_metrics.json")
    monkeypatch.setattr(drift, "DRIFT_REPORT_HTML", tmp_path / "report.html")

    metrics = drift.generate_drift_analysis(reference.sample(100, random_state=1))

    assert metrics["status"] == "STABLE"
    assert metrics["psi_text_length"] < 0.25
    saved = json.loads((tmp_path / "drift_metrics.json").read_text())
    assert saved["status"] == "STABLE"