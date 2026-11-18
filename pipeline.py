"""Retriever -> Generator pipeline (Nov milestone). Later the Orchestrator wraps this and adds Verifier/Critic.

Usage:
  python pipeline.py "अंगणवाडी सेविकांना मानधन किती मिळते?"
  python pipeline.py "What is the honorarium of Anganwadi workers?" --model llama-3.1-8b
  python pipeline.py --chat            (interactive)
  python pipeline.py "..." --mock      (no LLM; only tests the plumbing)
"""
import argparse, json, time
from retriever_agent import RetrieverAgent
from generator_agent import GeneratorAgent
from nov_config import DEFAULT_MODEL, TOP_K_EVIDENCE, EVIDENCE_LANG, LOGS_DIR


class GroundedPipeline:
    def __init__(self, model_key=DEFAULT_MODEL, mock=False):
        self.retriever = RetrieverAgent()
        self.generator = GeneratorAgent(model_key=model_key, mock=mock)

    def ask(self, query, k=TOP_K_EVIDENCE, lang=EVIDENCE_LANG, log=True):
        t0 = time.perf_counter()
        r = self.retriever.retrieve(query, k=k, lang=lang)
        g = self.generator.generate(query, r["evidence"])
        res = {"query": query,
               "answer": g["answer"], "abstained": g["abstained"],
               "generator_confidence": g["confidence"], "retrieval_top_score": r["top_score"],
               "sources": [{"rank": e["rank"], "chunk_id": e["chunk_id"], "doc_id": e["doc_id"],
                            "department": e["department"], "pages": e["pages"], "score": e["score"]}
                           for e in r["evidence"]],
               "cited_chunk_ids": g["cited_chunk_ids"], "model": g["model"],
               "timing_ms": {"retrieval": r["latency_ms"], "generation": g["latency_ms"],
                             "total": round((time.perf_counter() - t0) * 1000, 1)}}
        if log:   # every run is saved -> raw material for thesis tables
            LOGS_DIR.mkdir(exist_ok=True)
            with open(LOGS_DIR / "runs.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps({**res, "evidence_text": [e["text"] for e in r["evidence"]]},
                                   ensure_ascii=False) + "\n")
        return res


def show(res):
    print("\n" + "=" * 70)
    print("Q:", res["query"])
    print("A:", res["answer"])
    print(f"\nabstained={res['abstained']}  confidence={res['generator_confidence']}  "
          f"retrieval_top_score={res['retrieval_top_score']}  model={res['model']}")
    print("timing_ms:", res["timing_ms"])
    print("sources:")
    for s in res["sources"]:
        print(f"  [{s['rank']}] {s['chunk_id']}  pages={s['pages']}  score={s['score']}  ({s['department'][:25]})")
    print("=" * 70)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("query", nargs="?"); ap.add_argument("--chat", action="store_true")
    ap.add_argument("--model", default=DEFAULT_MODEL); ap.add_argument("--mock", action="store_true")
    ap.add_argument("-k", type=int, default=TOP_K_EVIDENCE)
    a = ap.parse_args()
    p = GroundedPipeline(a.model, a.mock)
    if a.chat:
        while (q := input("\nप्रश्न / Question (exit to quit): ").strip()) and q.lower() != "exit":
            show(p.ask(q, k=a.k))
    elif a.query:
        show(p.ask(a.query, k=a.k))
    else:
        ap.error("give a query or --chat")
