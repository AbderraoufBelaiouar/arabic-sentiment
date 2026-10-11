"""
Shared test fixtures — reusable objects for all test files.

Session-scoped fixtures load heavy resources (model, API client) once
for ALL tests rather than per-test.
"""

import pytest
from fastapi.testclient import TestClient

from arabic_sentiment.api.main import app
from arabic_sentiment.model import SentimentClassifier


@pytest.fixture(scope="session")
def classifier():
    """Load the classifier once for all tests."""
    clf = SentimentClassifier()
    clf.load()
    return clf


@pytest.fixture(scope="session")
def client():
    """Create a test client for the API."""
    with TestClient(app) as c:
        yield c


@pytest.fixture
def sample_arabic_texts():
    """Sample texts for testing."""
    return [
        ("المنتج رائع والتوصيل سريع", "positive"),
        ("جودة رديئة ولن أشتري مرة أخرى", "negative"),
        ("المنتج عادي", "neutral"),
    ]
