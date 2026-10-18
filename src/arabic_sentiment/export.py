"""
Script to export the trained PyTorch model to ONNX format for serving.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from arabic_sentiment.config import MODELS_DIR, settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_SOURCE_DIR = MODELS_DIR / "arabert-twitter-optimized"


def export_model_to_onnx(
    source_dir: str | Path | None = None,
    target_dir: str | Path | None = None,
) -> None:
    """Export the PyTorch checkpoint in source_dir to ONNX in target_dir."""
    source = Path(source_dir) if source_dir else DEFAULT_SOURCE_DIR
    target = Path(target_dir) if target_dir else Path(settings.model_path)
    output_path = target / "model.onnx"

    logger.info(f"Loading PyTorch model from {source}...")
    model = AutoModelForSequenceClassification.from_pretrained(source)
    tokenizer = AutoTokenizer.from_pretrained(source)

    # Put the model in evaluation mode
    model.eval()

    logger.info("Creating dummy input for tracing...")
    dummy_text = "هذا مجرد اختبار"
    inputs = tokenizer(
        dummy_text,
        return_tensors="pt",
        truncation=True,
        max_length=settings.max_seq_length,
        padding="max_length",
    )

    target.mkdir(parents=True, exist_ok=True)
    for stale in (output_path, Path(f"{output_path}.data")):
        stale.unlink(missing_ok=True)

    logger.info(f"Exporting ONNX model to {output_path}...")
    torch.onnx.export(
        model,
        (inputs["input_ids"], inputs["attention_mask"]),
        str(output_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "logits": {0: "batch_size"},
        },
        opset_version=14,
        do_constant_folding=True,
    )
    tokenizer.save_pretrained(target)
    logger.info("Export successful!")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Arabic Sentiment model to ONNX")
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help=(
            "Directory containing the trained PyTorch model. "
            "Defaults to models/arabert-twitter-optimized."
        ),
    )
    parser.add_argument(
        "--target",
        type=str,
        default=None,
        help=(
            "Directory to write the ONNX model and tokenizer to. "
            "Defaults to settings.model_path (models/arabert-twitter-onnx)."
        ),
    )
    args = parser.parse_args()
    export_model_to_onnx(args.source, args.target)


if __name__ == "__main__":
    main()
