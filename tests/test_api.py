"""Test the API endpoints."""


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
