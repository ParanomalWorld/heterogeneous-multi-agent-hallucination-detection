"""End-to-end evaluation with the REAL local LLM (Mar milestone) -> real-world tables of the paper.

Step 1  every query in eval_queries.json is answered ONCE by the full pipeline (generation is the slow part),
Step 2  the cached answers are re-verified under every policy (cheap) -> fair policy comparison on real answers,
Step 3  writes results/e2e_annotation.csv: one row per claim with the evidence. Fill the column human_label
        (SUPPORTED / NUMERIC_FABRICATION / CONTRADICTION / UNSUPPORTED) yourself or with a classmate,
        then run  `python evaluate_e2e.py --score results/e2e_annotation.csv`  -> agreement / precision / recall.

Usage:  python evaluate_e2e.py [--model llama-3.2-3b] [--budget 90] [--no-tier2]
        python evaluate_e2e.py --mock            (plumbing test)
"""
import argparse, csv, json, statistics as st
from collections import Counter
from pathlib import Path

from nov_config import DEFAULT_MODEL, ROOT
from hallu_config import POLICIES, LATENCY_BUDGET_S
from orchestrator import Orchestrator

RESULTS = ROOT / "results"


class CachedGenerator:
    def __init__(self):
        self.cache = {}
    def generate(self, query, evidence, on_token=None, **kw):
        return self.cache[query]


def score(path):
    rows = [r for r in csv.DictReader(open(path, encoding="utf-8-sig")) if r.get("human_label", "").strip()]
    if not rows:
        print("No human_label filled in yet."); return
    flag = lambda t: t.strip().upper() not in ("SUPPORTED", "")
    tp = sum(flag(r["system_error_type"]) and flag(r["human_label"]) for r in rows)
    fp = sum(flag(r["system_error_type"]) and not flag(r["human_label"]) for r in rows)
    fn = sum(not flag(r["system_error_type"]) and flag(r["human_label"]) for r in rows)
    agree = sum(r["system_error_type"].strip().upper() == r["human_label"].strip().upper() for r in rows)
    p = tp / (tp + fp) if tp + fp else 0; rc = tp / (tp + fn) if tp + fn else 0
    out = {"n_claims": len(rows), "human_hallucination_rate": round(sum(flag(r["human_label"]) for r in rows) / len(rows), 3),
           "detector_precision": round(p, 3), "detector_recall": round(rc, 3),
           "detector_f1": round(2 * p * rc / (p + rc), 3) if p + rc else 0, "exact_type_agreement": round(agree / len(rows), 3)}
    print(json.dumps(out, indent=2)); json.dump(out, open(RESULTS / "e2e_human_scores.json", "w"), indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL); ap.add_argument("--mock", action="store_true")
    ap.add_argument("--budget", type=float, default=LATENCY_BUDGET_S); ap.add_argument("--score")
    ap.add_argument("--no-tier2", action="store_true")
    a = ap.parse_args()
    RESULTS.mkdir(exist_ok=True)
    if a.score:
        return score(a.score)
    qs = json.load(open(ROOT / "eval_queries.json", encoding="utf-8"))
    live = Orchestrator(a.model, mock_llm=a.mock, mock_nli=a.mock, load_tier2=not a.no_tier2)
    cache = CachedGenerator()
    # ---------- step 1: real generation once (adaptive policy) ----------
    runs = {}
    for q in qs:
        captured = {}
        orig = live.generator.generate
        def spy(query, evidence, **kw):
            g = orig(query, evidence, **kw); captured["g"] = g; return g
        live.generator.generate = spy
        r = live.run(q["query"], policy="adaptive", budget_s=a.budget)
        live.generator.generate = orig
        if "g" in captured:
            cache.cache[q["query"]] = captured["g"]
        runs[q["id"]] = r
        print(f"#{q['id']:>2} {r['verdict']:<20} {r['timing_ms']['total']/1000:6.1f}s  {q['query'][:50]}")
    # ---------- step 2: same answers, every policy ----------
    off = Orchestrator(retriever=live.retriever, generator=cache, mock_nli=a.mock, load_tier2=not a.no_tier2)
    off.verifier._t1, off.verifier._t2 = live.verifier._t1, live.verifier._t2      # reuse loaded NLI models
    table = []
    for pol in POLICIES:
        res = [off.run(q["query"], policy=pol, budget_s=None, log=False) if q["query"] in cache.cache else runs[q["id"]]
               for q in qs]
        ins = [r for r, q in zip(res, qs) if q["in_scope"]]
        claims = [c for r in res for c in r["claims"]]
        table.append({"policy": pol, "verdicts": dict(Counter(r["verdict"] for r in res)),
                      "claim_flag_rate": round(sum(c["error_type"] not in ("SUPPORTED", "CITATION_MISMATCH")
                                                   for c in claims) / max(1, len(claims)), 3),
                      "mean_verify_ms": round(st.mean(r["timing_ms"].get("verification", 0) for r in res), 1),
                      "mean_faithfulness_in_scope": round(st.mean([r["faithfulness"] for r in ins
                                                                   if r["faithfulness"] is not None] or [0]), 3)})
    # ---------- summary ----------
    R = list(runs.values())
    ins = [runs[q["id"]] for q in qs if q["in_scope"]]; outs = [runs[q["id"]] for q in qs if not q["in_scope"]]
    ood_scores, in_scores = [r["retrieval_top_score"] for r in outs], [r["retrieval_top_score"] for r in ins]
    stage = lambda k: round(st.mean(r["timing_ms"].get(k, 0) for r in R), 1)
    summary = {"model": a.model if not a.mock else "mock", "n_queries": len(R),
               "latency_ms_mean": {k: stage(k) for k in ("retrieval", "generation", "precheck", "verification", "critic", "total")},
               "verdicts_live_adaptive": dict(Counter(r["verdict"] for r in R)),
               "abstain_rate_in_scope": round(sum(r["abstained"] for r in ins) / max(1, len(ins)), 3),
               "abstain_rate_out_of_scope": round(sum(r["abstained"] for r in outs) / max(1, len(outs)), 3),
               "retrieval_top_score": {"in_scope_min": min(in_scores, default=None), "out_scope_max": max(ood_scores, default=None)},
               "suggested_EARLY_EXIT_T": (round((max(ood_scores) + min(in_scores)) / 2, 3)
                                          if ood_scores and in_scores and max(ood_scores) < min(in_scores) else
                                          "not separable - keep 0 (off)"),
               "policies_on_same_answers": table}
    json.dump(summary, open(RESULTS / "e2e_summary.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    with open(RESULTS / "e2e_runs.jsonl", "w", encoding="utf-8") as f:
        for r in R:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(RESULTS / "e2e_annotation.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["query_id", "query", "claim", "evidence_premise", "system_error_type", "system_reason", "human_label"])
        for q in qs:
            for c in runs[q["id"]]["claims"]:
                w.writerow([q["id"], q["query"], c["text"], " | ".join(p["text"] for p in c["premise"]),
                            c["error_type"], c["reason"], ""])
    print("\nSUMMARY:", json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nNext: open {RESULTS / 'e2e_annotation.csv'} in Excel, fill human_label, then run --score")


if __name__ == "__main__":
    main()
