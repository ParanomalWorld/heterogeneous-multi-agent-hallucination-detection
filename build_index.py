"""Step 1: load chunks -> clean/validate -> embed -> build local FAISS index.

Usage:  python build_index.py            (real model, needs HF model cached/downloaded once)
        python build_index.py --dummy    (pipeline smoke test, no model)
"""
import argparse, json, time
import faiss, numpy as np
from collections import Counter
from common import CHUNKS_PATH, INDEX_DIR, MODEL_NAME, MIN_WORDS, clean_text, get_embedder


def load_and_clean(path):
    raw = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    kept, seen, stats = [], set(), Counter()
    for r in raw:
        r["text"] = clean_text(r["text"])
        r["n_words"] = len(r["text"].split())
        if r["n_words"] < MIN_WORDS:
            stats["dropped_short"] += 1; continue
        key = (r["lang"], r["text"])
        if key in seen:
            stats["dropped_duplicate"] += 1; continue
        seen.add(key); kept.append(r)
    stats["input"], stats["kept"] = len(raw), len(kept)
    return kept, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dummy", action="store_true")
    ap.add_argument("--chunks", default=str(CHUNKS_PATH))
    a = ap.parse_args()

    rows, stats = load_and_clean(a.chunks)
    print("Cleaning:", dict(stats), "| by lang:", dict(Counter(r["lang"] for r in rows)))

    emb = get_embedder(a.dummy)
    t0 = time.time()
    X = emb.embed_passages([r["text"] for r in rows])
    print(f"Embedded {X.shape} in {time.time()-t0:.1f}s")

    index = faiss.IndexFlatIP(X.shape[1])   # exact cosine (vectors are L2-normalised); 6k chunks is tiny
    index.add(X)

    INDEX_DIR.mkdir(exist_ok=True)
    faiss.write_index(index, str(INDEX_DIR / "policy.faiss"))
    with open(INDEX_DIR / "metadata.jsonl", "w", encoding="utf-8") as f:   # row i == FAISS id i
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    json.dump({"model": "dummy" if a.dummy else MODEL_NAME, "dim": int(X.shape[1]),
               "n_vectors": int(index.ntotal), "cleaning": dict(stats)},
              open(INDEX_DIR / "build_info.json", "w"), indent=2)
    print(f"Saved index with {index.ntotal} vectors -> {INDEX_DIR}")


if __name__ == "__main__":
    main()
