"""Shared helpers: config, text normalisation, embedder (real + dummy for smoke tests)."""
import hashlib, re, unicodedata
from pathlib import Path
import numpy as np

ROOT = Path(__file__).parent
CHUNKS_PATH = ROOT / "data" / "chunks.jsonl"
INDEX_DIR = ROOT / "index"
# Multilingual (supports Marathi + English, cross-lingual), small enough for CPU.
# Alternatives: "intfloat/multilingual-e5-base" (better, slower), "BAAI/bge-m3" (best, heavy).
MODEL_NAME = "intfloat/multilingual-e5-small"
MIN_WORDS = 20          # chunks shorter than this are dropped as noise (headers, page stubs)

_WS = re.compile(r"\s+")
_DOTS = re.compile(r"\.{4,}")
_DASHES = re.compile(r"-{4,}")          # table rules / page separators from PDF extraction


def clean_text(t: str) -> str:
    """Light, safe cleaning: NFC unicode, drop zero-width chars, collapse dots/whitespace."""
    t = unicodedata.normalize("NFC", t)
    t = t.replace("\u200b", "").replace("\u200c", "").replace("\u200d", "").replace("\ufeff", "")
    t = _DOTS.sub("...", t)
    t = _DASHES.sub(" ", t)
    return _WS.sub(" ", t).strip()


class E5Embedder:
    """sentence-transformers wrapper. E5 models need 'query: ' / 'passage: ' prefixes."""
    def __init__(self, name=MODEL_NAME):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(name, device="cpu")
        self.dim = self.model.get_sentence_embedding_dimension()

    def _enc(self, texts, bs=32, bar=False):
        return self.model.encode(texts, batch_size=bs, normalize_embeddings=True,
                                 show_progress_bar=bar).astype("float32")

    def embed_passages(self, texts, bar=True):
        return self._enc([f"passage: {t}" for t in texts], bar=bar)

    def embed_query(self, text):
        return self._enc([f"query: {text}"])


class DummyEmbedder:
    """Hashed bag-of-words (NO semantics). Only for smoke-testing the pipeline offline."""
    dim = 256
    def _vec(self, t):
        v = np.zeros(self.dim, "float32")
        for w in re.findall(r"\w+", t.lower()):
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % self.dim] += 1
        n = np.linalg.norm(v)
        return v / n if n else v
    def embed_passages(self, texts, bar=False):
        return np.stack([self._vec(t) for t in texts])
    def embed_query(self, text):
        return self._vec(text)[None]


def get_embedder(dummy=False):
    return DummyEmbedder() if dummy else E5Embedder()
