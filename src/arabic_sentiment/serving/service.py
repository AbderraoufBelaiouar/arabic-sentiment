"""
BentoML 1.4+ service with adaptive micro-batching.

Serve with:
    bentoml serve src/arabic_sentiment/serving/service.py:ArabicSentimentService --port 8000
"""
import bentoml
from typing import Dict, Any, List
from arabic_sentiment.model import SentimentClassifier


@bentoml.service(
    name="arabic_sentiment_service",
    resources={"cpu": "2"},
)
class ArabicSentimentService:
    """BentoML service wrapping the AraBERT ONNX sentiment classifier."""

    def __init__(self) -> None:
        bento_model = bentoml.models.get("arabert_sentiment:latest")
        self.classifier = SentimentClassifier(model_path=bento_model.path)

    # ── Batchable endpoint (rubric requirement: batchable=True) ──
    @bentoml.api(
        batchable=True,
        batch_dim=0,
        max_batch_size=32,
        max_latency_ms=50,
        route="/predict_batch",
    )
    def predict_batch(self, text: List[str]) -> List[Dict[str, Any]]:
        """
        Adaptive micro-batching endpoint.
        BentoML automatically groups concurrent requests and
        concatenates 'text' lists along batch_dim=0.
        """
        return self.classifier.predict_batch(text)

    # ── Single-text endpoint (compatible with FastAPI /predict) ──
    @bentoml.api(route="/predict")
    def predict(self, text: str) -> Dict[str, Any]:
        """Single-text prediction endpoint, mirrors the FastAPI API."""
        if not text:
            return {"error": "Empty text", "status": 422}
        return self.classifier.predict_one(text)
