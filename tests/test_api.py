"""Test the API endpoints."""
import re


def test_predict_batch_returns_results(client):
    """Batch endpoint should return one prediction per input text."""
    response = client.post(
        "/predict/batch",
        json=[{"text": "المنتج رائع"}, {"text": "جودة رديئة"}],
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2
    for item in data:
        assert item["label"] in ["positive", "negative", "neutral"]


def test_predict_updates_p95_gauge_with_real_value(client):
    """p95 gauge must reflect observed latencies, not a hardcoded constant."""
    client.post("/predict", json={"text": "المنتج رائع"})
    content = client.get("/metrics").text
    match = re.search(r"^arabic_sentiment_p95_latency_ms ([0-9.eE+-]+)$", content, re.MULTILINE)
    assert match is not None
    value = float(match.group(1))
    assert value > 0
    assert value != 980.0


def test_predict_failure_returns_500(client, monkeypatch):
    """Internal prediction errors must surface as 500, not crash the API."""

    def boom(_text):
        raise RuntimeError("simulated failure")

    monkeypatch.setattr(client.app.state.classifier, "predict_one", boom)
    response = client.post("/predict", json={"text": "المنتج رائع"})
    assert response.status_code == 500


def test_health_returns_503_when_model_not_loaded(client, monkeypatch):
    """Health must report 503 if the model is not loaded."""
    monkeypatch.setattr(client.app.state.classifier, "_loaded", False)
    response = client.get("/health")
    assert response.status_code == 503
def test_health_returns_200(client):
    """Health endpoint should return 200 when model is loaded."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_predict_returns_valid_label(client):
    """Predict endpoint should return a valid sentiment label."""
    response = client.post("/predict", json={"text": "المنتج رائع"})
    assert response.status_code == 200
    data = response.json()
    assert data["label"] in ["positive", "negative", "neutral"]
    assert 0 <= data["confidence"] <= 1


def test_predict_empty_text_returns_422(client):
    """Empty text should be rejected with 422."""
    response = client.post("/predict", json={"text": ""})
    assert response.status_code == 422


def test_predict_missing_text_returns_422(client):
    """Missing text field should be rejected with 422."""
    response = client.post("/predict", json={})
    assert response.status_code == 422


def test_metadata_returns_200(client):
    """Metadata endpoint should return model info."""
    response = client.get("/metadata")
    assert response.status_code == 200
    data = response.json()
    assert "model_version" in data
    assert "model_name" in data
    assert data["framework"] in ["onnx", "pytorch"]


def test_metrics_endpoint_returns_prometheus_metrics(client):
    """Metrics endpoint should return 200 with Prometheus metrics."""
    response = client.get("/metrics")
    assert response.status_code == 200
    content = response.text
    assert "arabic_sentiment_psi_score" in content
    assert "arabic_sentiment_p95_latency_ms" in content
