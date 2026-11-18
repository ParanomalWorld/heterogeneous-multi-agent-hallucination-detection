"""Robustness to query variation: same information need, asked in different ways.
Styles: original | mr_paraphrase | en_paraphrase | romanized (Marathi in English letters) | keywords | spelling_variant
Reports Hit@1 / Hit@3 / MRR per style (retrieval decides everything downstream) and top-score of out-of-scope queries.

Usage:  python eval_paraphrase.py              (retrieval only, ~1 min)
        python eval_paraphrase.py --full       (also runs the 5-agent pipeline: verdict per query, slow)
Output: results/paraphrase_summary.json, results/paraphrase_table.md
"""
import argparse, json, statistics as st
from collections import defaultdict
from nov_config import ROOT, EVIDENCE_LANG
from retriever_agent import RetrieverAgent

ap = argparse.ArgumentParser(); ap.add_argument("--full", action="store_true"); ap.add_argument("--no-tier2", action="store_true")
ap.add_argument("--mock", action="store_true", help="mock LLM+NLI (plumbing test)")
a = ap.parse_args()
orig = [{**q, "style": "original"} for q in json.load(open(ROOT / "eval_queries.json", encoding="utf-8"))]
var = json.load(open(ROOT / "eval_paraphrase.json", encoding="utf-8"))
qs = orig + var
R = RetrieverAgent()
orch = None
if a.full:
    from orchestrator import Orchestrator
    orch = Orchestrator(retriever=R, load_tier2=not a.no_tier2, mock_llm=a.mock, mock_nli=a.mock)

rows = []
for q in qs:
    r = R.retrieve(q["query"], k=5, lang=EVIDENCE_LANG)
    docs = [e["doc_id"] for e in r["evidence"]]
    rank = docs.index(q["expected_doc_id"]) + 1 if q.get("expected_doc_id") in docs else None
    row = {"style": q["style"], "query": q["query"], "in_scope": q["in_scope"], "rank": rank, "top_score": r["top_score"]}
    if orch:
        res = orch.run(q["query"], log=False)
        row.update(verdict=res["verdict"], abstained=res["abstained"], total_s=round(res["timing_ms"]["total"] / 1000, 1))
    rows.append(row)
    print(f"{q['style']:<17} rank={str(rank):<5} score={r['top_score']:.3f} {row.get('verdict', ''):<18} {q['query'][:55]}")

by = defaultdict(list)
for r in rows:
    by[r["style"] if r["in_scope"] else "out_of_scope"].append(r)
summary = {}
for s, rs in by.items():
    if s == "out_of_scope":
        summary[s] = {"n": len(rs), "mean_top_score": round(st.mean(r["top_score"] for r in rs), 3),
                      "max_top_score": round(max(r["top_score"] for r in rs), 3)}
    else:
        summary[s] = {"n": len(rs), "hit@1": round(sum(r["rank"] == 1 for r in rs) / len(rs), 3),
                      "hit@3": round(sum(bool(r["rank"]) and r["rank"] <= 3 for r in rs) / len(rs), 3),
                      "MRR": round(sum(1 / r["rank"] for r in rs if r["rank"]) / len(rs), 3),
                      "mean_top_score": round(st.mean(r["top_score"] for r in rs), 3)}
    if orch:
        summary[s]["abstain_rate"] = round(sum(r["abstained"] for r in rs) / len(rs), 3)
(ROOT / "results").mkdir(exist_ok=True)
json.dump({"summary": summary, "rows": rows}, open(ROOT / "results" / "paraphrase_summary.json", "w", encoding="utf-8"),
          indent=2, ensure_ascii=False)
L = ["## Table D - Robustness to query variation (retrieval, k=5)", "",
     "| Query style | n | Hit@1 | Hit@3 | MRR | Mean top score |", "|---|---|---|---|---|---|"]
for s, v in summary.items():
    if s != "out_of_scope":
        L.append(f"| {s} | {v['n']} | {v['hit@1']} | {v['hit@3']} | {v['MRR']} | {v['mean_top_score']} |")
if "out_of_scope" in summary:
    v = summary["out_of_scope"]
    L.append(f"| out_of_scope | {v['n']} | - | - | - | {v['mean_top_score']} (max {v['max_top_score']}) |")
(ROOT / "results" / "paraphrase_table.md").write_text("\n".join(L) + "\n", encoding="utf-8")
print("\n" + "\n".join(L))
