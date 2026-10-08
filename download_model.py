"""One-time download of a quantised GGUF model (needs internet ONCE; afterwards fully offline).
Usage: python download_model.py                 -> default (llama-3.2-3b, ~2 GB)
       python download_model.py llama-3.1-8b    -> 8B (~4.9 GB)
"""
import sys
from huggingface_hub import hf_hub_download
from nov_config import MODEL_REGISTRY, DEFAULT_MODEL, MODELS_DIR

name = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL
spec = MODEL_REGISTRY[name]
MODELS_DIR.mkdir(exist_ok=True)
print(f"Downloading {spec['file']} (~{spec['size_gb']} GB) ...")
path = hf_hub_download(repo_id=spec["repo"], filename=spec["file"], local_dir=MODELS_DIR)
print("Saved:", path)
