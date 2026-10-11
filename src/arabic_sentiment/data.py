"""Data loading and preprocessing pipeline.

Responsible for:
1. Downloading the Arabic tweet sentiment dataset from HuggingFace.
2. Applying ArabertPreprocessor normalisation + BERT tokenisation.
"""

from __future__ import annotations

from arabert.preprocess import ArabertPreprocessor
from datasets import DatasetDict, load_from_disk
from transformers import PreTrainedTokenizerBase

from arabic_sentiment.config import (
    DATA_DIR,
    settings,
)


def load_arabic_sentiment_dataset() -> DatasetDict:
    """Load the local Arabic tweet-sentiment dataset."""
    return load_from_disk(str(DATA_DIR))


def build_preprocess_fn(
    preprocessor: ArabertPreprocessor,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int | None = None,
):
    """Return a batched preprocessing function for `dataset.map()`."""
    if max_length is None:
        max_length = settings.max_seq_length

    def _preprocess(examples: dict) -> dict:
        # Arabic-specific normalisation (diacritics, letter forms, …)
        texts = [preprocessor.preprocess(text) for text in examples["text"]]

        # Sub-word tokenisation
        result = tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )

        # Cast string labels → int
        if "label" in examples:
            result["label"] = [int(label) for label in examples["label"]]

        return result

    return _preprocess


def tokenize_dataset(
    dataset: DatasetDict,
    preprocessor: ArabertPreprocessor,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int | None = None,
) -> DatasetDict:
    """Apply ArabertPreprocessor + tokeniser to every split."""
    if max_length is None:
        max_length = settings.max_seq_length
    preprocess_fn = build_preprocess_fn(preprocessor, tokenizer, max_length)
    return dataset.map(preprocess_fn, batched=True)
