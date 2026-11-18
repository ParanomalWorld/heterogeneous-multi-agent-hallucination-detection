"""One-time download of the Verifier NLI models (internet needed ONCE; afterwards fully offline).
Usage: python download_nli.py            -> Tier-1 multilingual mDeBERTa-base (~1.1 GB)  [required]
       python download_nli.py --tier2    -> also Tier-2 DeBERTa-v3-large MNLI (~1.7 GB)   [16 GB RAM laptops]
Saved under models/nli/<model-name>/ ; verifier_agent.py loads from there automatically.
"""
import argparse, os
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ["HF_HUB_OFFLINE"] = "0"     # downloading needs the network even if run_app.ps1 set offline mode
from huggingface_hub import snapshot_download
from hallu_config import NLI_TIER1, NLI_TIER2, NLI_DIR

ap = argparse.ArgumentParser(); ap.add_argument("--tier2", action="store_true"); a = ap.parse_args()
NLI_DIR.mkdir(parents=True, exist_ok=True)
for name in [NLI_TIER1] + ([NLI_TIER2] if a.tier2 else []):
    print(f"Downloading {name} ...")
    p = snapshot_download(repo_id=name, local_dir=NLI_DIR / name.split("/")[-1],
                          allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "spm*"],
                          ignore_patterns=["*onnx*", "*.h5", "*.msgpack", "*.ot"])
    print("Saved:", p)
print("Test:  python verifier_agent.py")
