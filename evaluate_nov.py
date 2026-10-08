"""Evaluation for thesis tables. Run after the pipeline works.
 - Retrieval: Hit@1 / Hit@3 / MRR (is the expected GR among the retrieved docs?)
 - Generation: abstention on out-of-scope questions, confidence, latency, citations
Outputs: logs/eval_results.jsonl and logs/eval_summary.json.  Read the answers yourself too (manual faithfulness check).
Usage: python evaluate_nov.py [--model llama-3.2-3b] [--mock] [--no-gen]
"""
import argparse, json, statistics as st
from pipeline import GroundedPipeline
from nov_config import LOGS_DIR, DEFAULT_MODEL, ROOT

ap = argparse.ArgumentParser()
ap.add_argument("--model", default=DEFAULT_MODEL); ap.add_argument("--mock", action="store_true")
ap.add_argument("--no-gen", action="store_true", help="retrieval metrics only (fast)")
a = ap.parse_args()

qs = json.load(open(ROOT / "eval_queries.json", encoding="utf-8"))
p = GroundedPipeline(a.model, mock=a.mock or a.no_gen)
LOGS_DIR.mkdir(exist_ok=True)
rows = []
for q in qs:
    res = p.ask(q["query"], k=5, log=False) if a.no_gen else p.ask(q["query"], log=False)
    docs = [s["doc_id"] for s in res["sources"]]
    rank = docs.index(q["expected_doc_id"]) + 1 if q["expected_doc_id"] in docs else None
    rows.append({**q, "rank_of_expected": rank, "answer": res["answer"], "abstained": res["abstained"],
                 "confidence": res["generator_confidence"], "retrieval_top_score": res["retrieval_top_score"],
                 "timing_ms": res["timing_ms"], "model": res["model"]})
    print(f"#{q['id']:>2} rank={rank}  abstained={res['abstained']}  {q['query'][:50]}")

ins = [r for r in rows if r["in_scope"]]; outs = [r for r in rows if not r["in_scope"]]
summary = {"model": rows[0]["model"], "n_in_scope": len(ins), "n_out_of_scope": len(outs),
           "hit@1": round(sum(r["rank_of_expected"] == 1 for r in ins) / len(ins), 3),
           "hit@3": round(sum(bool(r["rank_of_expected"]) and r["rank_of_expected"] <= 3 for r in ins) / len(ins), 3),
           "MRR": round(sum(1 / r["rank_of_expected"] for r in ins if r["rank_of_expected"]) / len(ins), 3),
           "mean_retrieval_ms": round(st.mean(r["timing_ms"]["retrieval"] for r in rows), 1),
           "mean_generation_ms": round(st.mean(r["timing_ms"]["generation"] for r in rows), 1)}
if not (a.mock or a.no_gen):
    summary["abstain_rate_in_scope"] = round(sum(r["abstained"] for r in ins) / len(ins), 3)
    summary["abstain_rate_out_of_scope"] = round(sum(r["abstained"] for r in outs) / len(outs), 3)
with open(LOGS_DIR / "eval_results.jsonl", "w", encoding="utf-8") as f:
    for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
json.dump(summary, open(LOGS_DIR / "eval_summary.json", "w"), indent=2)
print("\nSUMMARY:", json.dumps(summary, indent=2))
