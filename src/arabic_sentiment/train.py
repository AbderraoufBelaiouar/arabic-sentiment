"""Training entrypoint for Arabic Sentiment Analysis.

Orchestrates: load → preprocess → train → eval → save.
Best config from notebook Phase 3: Twitter-AraBERT, cosine LR, weighted CE, early stopping.

Usage:
    python -m arabic_sentiment.train
    arabic-sentiment                    # via pyproject.toml entrypoint
"""

from __future__ import annotations

import json
import os

import mlflow
from transformers import EarlyStoppingCallback, TrainingArguments

from arabic_sentiment.config import (
    EARLY_STOPPING_PATIENCE,
    EVAL_BATCH_SIZE,
    LEARNING_RATE,
    LR_SCHEDULER,
    METRIC_FOR_BEST_MODEL,
    MODELS_DIR,
    NUM_EPOCHS,
    REPORTS_DIR,
    TRAIN_BATCH_SIZE,
    WARMUP_STEPS,
    WEIGHT_DECAY,
    settings,
)
from arabic_sentiment.data import load_arabic_sentiment_dataset, tokenize_dataset
from arabic_sentiment.metrics import compute_metrics
from arabic_sentiment.model import (
    WeightedTrainer,
    compute_class_weights,
    load_model_and_tokenizer,
)


def main() -> None:
    """Run the full training pipeline."""

    output_dir = str(MODELS_DIR / "arabert-twitter-optimized")

    # ── 1. Load dataset ─────────────────────────────
    print("📥  Loading dataset …")
    dataset = load_arabic_sentiment_dataset()
    print(
        f"    Train: {len(dataset['train'])}  |  "
        f"Test: {len(dataset['test'])}  |  "
        f"Val: {len(dataset['validation'])}"
    )

    # ── 2. Model / tokenizer / preprocessor ──────────
    print("🤖  Loading model & tokenizer …")
    model, tokenizer, preprocessor = load_model_and_tokenizer()
    print(f"    Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ── 3. Preprocess & tokenise ────────────────────
    print("⚙️   Tokenising dataset …")
    tokenized = tokenize_dataset(dataset, preprocessor, tokenizer)

    # ── 4. Class weights ────────────────────────────
    class_weights = compute_class_weights(tokenized["train"]["label"])
    print(f"⚖️   Class weights: {class_weights.cpu().numpy()}")

    # ── 5. Training arguments ───────────────────────
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type=LR_SCHEDULER,
        weight_decay=WEIGHT_DECAY,
        warmup_steps=WARMUP_STEPS,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model=METRIC_FOR_BEST_MODEL,
        report_to="none",
    )

    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        compute_metrics=compute_metrics,
        processing_class=tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE)],
    )
    mlflow.set_experiment("Arabic_Sentiment_Training")
    with mlflow.start_run():
    # Log your setup
        mlflow.log_params({
        "model_name": settings.model_name,
        "learning_rate": LEARNING_RATE,
        "batch_size": TRAIN_BATCH_SIZE,
        "epochs": NUM_EPOCHS
        })
    # ── 6. Train ────────────────────────────────────
        print("🏋️  Starting training …")
        trainer.train(resume_from_checkpoint=os.environ.get("RESUME_CHECKPOINT"))

    # ── 7. Evaluate ─────────────────────────────────
        print("📊  Evaluating …")
        val_results = trainer.evaluate()
        test_results = trainer.evaluate(eval_dataset=tokenized["test"])
        mlflow.log_metrics({
            "val_accuracy": val_results["eval_accuracy"],
            "val_f1_macro": val_results["eval_f1_macro"],
            "test_accuracy": test_results["eval_accuracy"],
            "test_f1_macro": test_results["eval_f1_macro"],
        })
        accuracy = test_results["eval_accuracy"]
        f1_macro = test_results["eval_f1_macro"]
        print(f"    Accuracy : {accuracy:.4f}")
        print(f"    F1 Macro : {f1_macro:.4f}")

    # ── 8. Save model ──────────────────────────────
    print(f"💾  Saving model → {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # ── 9. Save report ─────────────────────────────
    os.makedirs(REPORTS_DIR, exist_ok=True)
    report_path = REPORTS_DIR / "module-1.md"
    with open(report_path, "w") as f:
        f.write("# Module 1 Report\n\n")
        f.write("## Optimized Twitter-AraBERT Metrics\n\n")
        f.write(f"- **Accuracy:** {accuracy:.4f}\n")
        f.write(f"- **F1 Macro:** {f1_macro:.4f}\n")
        f.write(f"- **Model:** {settings.model_name}\n")
        f.write(f"- **Epochs:** {NUM_EPOCHS} (with Early Stopping)\n")
        f.write(f"- **Scheduler:** {LR_SCHEDULER}\n")

    metrics_path = REPORTS_DIR / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(
            {
                "accuracy": round(accuracy, 4),
                "f1_macro": round(f1_macro, 4),
                "model_name": settings.model_name,
                "epochs": NUM_EPOCHS,
                "scheduler": LR_SCHEDULER,
            },
            f,
            indent=2,
        )

    print(f"📝  Report saved → {report_path}")
    print("✅  Done!")


if __name__ == "__main__":
    main()
