"""
Script to export the PyTorch model to ONNX format.
"""

from __future__ import annotations

import argparse
import logging
import os

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from arabic_sentiment.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def export_model_to_onnx(output_path: str | None = None) -> None:
    if output_path is None:
        output_path = os.path.join(settings.model_path, "model.onnx")

    logger.info(f"Loading PyTorch model from {settings.model_path}...")
    model = AutoModelForSequenceClassification.from_pretrained(settings.model_path)
    tokenizer = AutoTokenizer.from_pretrained(settings.model_path)

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

    logger.info(f"Exporting ONNX model to {output_path}...")
    torch.onnx.export(
        model,
        (inputs["input_ids"], inputs["attention_mask"]),
        output_path,
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
    logger.info("Export successful!")

def main() -> None:
    parser = argparse.ArgumentParser(description="Export Arabic Sentiment model to ONNX")
    parser.add_argument(
        "--output", type=str, default=None, 
        help="Output path for the ONNX model. Defaults to model_path/model.onnx in settings."
    )
    args = parser.parse_args()
    export_model_to_onnx(args.output)

if __name__ == "__main__":
    main()
