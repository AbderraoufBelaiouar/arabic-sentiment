import bentoml
import shutil
from pathlib import Path
from arabic_sentiment.config import MODELS_DIR

def save_to_bentoml():
    onnx_dir = MODELS_DIR / "arabert-twitter-onnx"
    
    # Package the ONNX model directory as a versioned BentoML model
    with bentoml.models.create(
        name="arabert_sentiment",
        labels={"framework": "onnx", "task": "sentiment"},
        metadata={"model_name": "aubmindlab/bert-base-arabertv02-twitter"},
    ) as bento_model:
        # Copy the ONNX files and tokenizer into the BentoML model store path
        shutil.copytree(onnx_dir, bento_model.path, dirs_exist_ok=True)
    
    print(f"✅ Model saved to BentoML: {bento_model.tag}")

if __name__ == "__main__":
    save_to_bentoml()
