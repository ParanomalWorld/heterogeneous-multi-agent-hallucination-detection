"""November settings in one place (change here, not inside the agents)."""
from pathlib import Path

ROOT = Path(__file__).parent
MODELS_DIR = ROOT / "models"
LOGS_DIR = ROOT / "logs"

# GGUF = quantised (4-bit) model file format used by llama.cpp. Q4_K_M = best size/quality balance.
MODEL_REGISTRY = {
    "llama-3.2-3b": {"repo": "bartowski/Llama-3.2-3B-Instruct-GGUF",
                     "file": "Llama-3.2-3B-Instruct-Q4_K_M.gguf", "size_gb": 2.0},
    "llama-3.1-8b": {"repo": "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
                     "file": "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf", "size_gb": 4.9},
}
DEFAULT_MODEL = "llama-3.2-3b"      # 8 GB RAM laptop: start here; switch to "llama-3.1-8b" to try the synopsis model

# Generation settings
N_CTX = 4096            # context window. Marathi is token-heavy, so keep k small
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.1       # low = more faithful, less creative (we want grounded answers)
TOP_K_EVIDENCE = 3      # chunks given to the Generator
EVIDENCE_LANG = "mr"    # Marathi chunks are the original text; English chunks are poor machine translations
NOT_FOUND_TOKEN = "NOT_FOUND"
