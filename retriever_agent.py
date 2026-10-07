"""Retriever Agent: query (English or Marathi) -> top-k evidence chunks from local FAISS index.
Output dict is what the Generator / Verifier agents will consume later (Nov+).

Usage:  python retriever_agent.py "महिला व बालविकास योजनेसाठी पात्रता काय आहे?" -k 5
        python retriever_agent.py "eligibility for Birsa Munda Krishi Kranti Yojana" --lang en
"""
import argparse, json, time
import faiss
from common import INDEX_DIR, get_embedder


class RetrieverAgent:
    def __init__(self, index_dir=INDEX_DIR, dummy=None):
        info = json.load(open(index_dir / "build_info.json"))
        self.index = faiss.read_index(str(index_dir / "policy.faiss"))
        self.meta = [json.loads(l) for l in open(index_dir / "metadata.jsonl", encoding="utf-8")]
        self.emb = get_embedder(dummy if dummy is not None else info["model"] == "dummy")

    def retrieve(self, query, k=5, lang=None, department=None, min_score=0.0):
        """lang: 'en' | 'mr' | None (both; cross-lingual). Returns evidence + timing for the latency monitor."""
        t0 = time.perf_counter()
        q = self.emb.embed_query(query)
        fetch = k if not (lang or department) else min(self.index.ntotal, k * 20)  # over-fetch, then filter
        scores, ids = self.index.search(q, fetch)
        ev = []
        for s, i in zip(scores[0], ids[0]):
            if i < 0 or s < min_score:
                continue
            m = self.meta[i]
            if lang and m["lang"] != lang: continue
            if department and m["department"] != department: continue
            ev.append({"rank": len(ev) + 1, "score": round(float(s), 4), "chunk_id": m["chunk_id"],
                       "doc_id": m["doc_id"], "lang": m["lang"], "department": m["department"],
                       "gr_no": m["gr_no"], "pages": [m["page_start"], m["page_end"]], "text": m["text"]})
            if len(ev) == k: break
        return {"query": query, "k": k, "filters": {"lang": lang, "department": department},
                "evidence": ev, "top_score": ev[0]["score"] if ev else 0.0,
                "latency_ms": round((time.perf_counter() - t0) * 1000, 1)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("query"); ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--lang", choices=["en", "mr"]); ap.add_argument("--dept")
    a = ap.parse_args()
    out = RetrieverAgent().retrieve(a.query, a.k, a.lang, a.dept)
    print(json.dumps(out, ensure_ascii=False, indent=2))
