"""Smoke tests for the config module."""

from arabic_sentiment.config import (
    LABEL_NAMES,
    LEARNING_RATE,
    NUM_EPOCHS,
    NUM_LABELS,
    settings,
)


def test_label_count_matches():
    assert NUM_LABELS == len(LABEL_NAMES)


def test_label_names_content():
    assert LABEL_NAMES == ["negative", "neutral", "positive"]


def test_model_name_is_twitter_variant():
    assert "twitter" in settings.model_name.lower()


def test_max_length_is_positive():
    assert settings.max_seq_length > 0


def test_num_epochs_is_positive():
    assert NUM_EPOCHS > 0


def test_learning_rate_range():
    assert 0 < LEARNING_RATE < 1


def test_settings_api_port():
    assert settings.api_port == 8000


def test_settings_model_version():
    assert settings.model_version == "0.1.0"
