"""MH-PolicyHalluBench: controlled hallucination benchmark built from the GR corpus itself + policy ablation.
This produces the MAIN RESULT TABLES of the paper (no LLM needed -> fast and fully reproducible).

How items are built (from index/metadata.jsonl):
  for a sampled GR chunk (the evidence), one sentence of it is a FAITHFUL answer (label 0), and
  perturbed copies are HALLUCINATED answers (label 1) with a known type:
     NUM_SWAP       a number changed                         -> expected NUMERIC_FABRICATION
     NEGATION       meaning flipped (Marathi/English rules)  -> expected CONTRADICTION
     CROSS_DOC      a sentence from another GR               -> expected UNSUPPORTED
     FABRICATED     an invented but plausible policy rule    -> expected UNSUPPORTED
Every item is run through the REAL orchestrator graph (precheck -> verify -> critic) with each policy:
  adaptive (proposed) | always_full | heuristic_only    (+ a latency-budget sweep for adaptive)

Usage:
  python hallu_bench.py --n 150                 (real NLI models)
  python hallu_bench.py --n 40 --mock-nli       (quick plumbing test)
Outputs: results/bench_items.jsonl, results/bench_summary.json, results/bench_tables.md, results/*.png
"""
import argparse, json, random, re, statistics as st, time
from pathlib import Path

from common import INDEX_DIR
from text_utils import split_sentences, numbers, tokens, DEV_DIGITS
from orchestrator import Orchestrator

RESULTS = Path(__file__).parent / "results"
FLAG_TYPES = {"NUMERIC_FABRICATION", "CONTRADICTION", "UNSUPPORTED", "UNVERIFIABLE"}   # UNCHECKED = not detected
EXPECTED = {"FAITHFUL": "SUPPORTED", "NUM_SWAP": "NUMERIC_FABRICATION", "NEGATION": "CONTRADICTION",
            "CROSS_DOC": "UNSUPPORTED", "FABRICATED": "UNSUPPORTED"}

NEG_RULES = [("देण्यात येईल", "देण्यात येणार नाही"), ("करण्यात येईल", "करण्यात येणार नाही"),
             ("अनुज्ञेय राहील", "अनुज्ञेय राहणार नाही"), ("लागू राहील", "लागू राहणार नाही"),
             ("आवश्यक आहे", "आवश्यक नाही"), ("पात्र राहतील", "पात्र राहणार नाहीत"), ("पात्र असतील", "पात्र नसतील"),
             ("मान्यता देण्यात येत आहे", "मान्यता नाकारण्यात येत आहे"), ("राहील", "राहणार नाही"),
             ("येईल", "येणार नाही"), (" will be ", " will not be "), (" is ", " is not "), (" shall ", " shall not ")]
FABRICATIONS_MR = ["या योजनेचा लाभ घेण्यासाठी अर्जदाराचे किमान पदवीपर्यंत शिक्षण पूर्ण झालेले असणे आवश्यक आहे.",
                   "या योजनेअंतर्गत लाभार्थ्यांना दरमहा रु. 2,500 अतिरिक्त प्रवास भत्ता देण्यात येईल.",
                   "अर्ज केवळ जिल्हाधिकारी कार्यालयात प्रत्यक्ष सादर करणे बंधनकारक राहील.",
                   "ही योजना फक्त मुंबई व पुणे महानगरपालिका क्षेत्रातच लागू राहील.",
                   "लाभार्थ्याची निवड ऑनलाईन लॉटरी पद्धतीने दरवर्षी 15 ऑगस्ट रोजी करण्यात येईल."]
REV = {v: k for k, v in zip("0123456789", "०१२३४५६७८९")}


def indian(n: int) -> str:
    s = str(n)
    if len(s) <= 3:
        return s
    head, tail = s[:-3], s[-3:]
    head = re.sub(r"(\d)(?=(\d{2})+$)", r"\1,", head)
    return f"{head},{tail}"


def num_swap(sent, rng):
    cands = [m for m in re.finditer(r"[\d०-९][\d०-९,]*", sent) if m.group().strip(",")]
    cands = [m for m in cands if int(m.group().translate(DEV_DIGITS).replace(",", "") or 0) >= 2]
    if not cands:
        return None
    m = rng.choice(cands)
    raw = m.group().rstrip(","); dev = any(ch in "०१२३४५६७८९" for ch in raw)
    v = int(raw.translate(DEV_DIGITS).replace(",", ""))
    nv = v * 2 if v < 10 else (v + rng.choice([1, 2, 3]) if 1900 <= v <= 2100 else int(round(v * rng.choice([1.5, 2, 0.5]))))
    if nv == v:
        nv = v + 1
    out = indian(nv) if "," in raw else str(nv)
    if dev:
        out = "".join(REV.get(ch, ch) if ch.isdigit() else ch for ch in out)
    return sent[:m.start()] + out + sent[m.start() + len(raw):]


def negate(sent):
    for a, b in NEG_RULES:
        if a in sent:
            return sent.replace(a, b, 1)
    return None


class FixedRetriever:
    def __init__(self):
        self.ev = []
    def retrieve(self, query, k=3, lang=None, **kw):
        return {"query": query, "evidence": self.ev, "top_score": 0.9, "latency_ms": 0.0}


class FixedGenerator:
    def __init__(self):
        self.answer = ""
    def generate(self, query, evidence, on_token=None, **kw):
        return {"answer": self.answer, "abstained": False, "confidence": None, "used_evidence": evidence,
                "cited_chunk_ids": [], "model": "fixed", "latency_ms": 0.0}


def build_items(n, seed):
    rng = random.Random(seed)
    meta = [json.loads(l) for l in open(INDEX_DIR / "metadata.jsonl", encoding="utf-8")]
    meta = [m for m in meta if m["lang"] == "mr"] or meta
    rng.shuffle(meta)
    items = []
    for m in meta:
        if len(items) >= n * 5:
            break
        sents = [s for s in split_sentences(m["text"]) if 6 <= len(tokens(s)) <= 45]
        if not sents:
            continue
        s = rng.choice(sents)
        other = rng.choice([x for x in meta if x["doc_id"] != m["doc_id"]] or meta)
        distract = {"chunk_id": other["chunk_id"], "text": other["text"]}
        evidence = [{"chunk_id": m["chunk_id"], "text": m["text"]}, distract]
        base = {"chunk_id": m["chunk_id"], "doc_id": m["doc_id"], "evidence": evidence, "source_sentence": s}
        variants = {"FAITHFUL": s, "NUM_SWAP": num_swap(s, rng), "NEGATION": negate(s),
                    "FABRICATED": rng.choice(FABRICATIONS_MR)}
        osents = [x for x in split_sentences(other["text"]) if 6 <= len(tokens(x)) <= 45]
        variants["CROSS_DOC"] = rng.choice(osents) if osents else None
        for typ, ans in variants.items():
            if ans and (typ == "FAITHFUL" or ans != s):
                items.append({**base, "id": len(items), "type": typ, "label": int(typ != "FAITHFUL"),
                              "answer": ans + " [1]"})
    return items


def prf(rows):
    tp = sum(r["pred"] and r["label"] for r in rows); fp = sum(r["pred"] and not r["label"] for r in rows)
    fn = sum((not r["pred"]) and r["label"] for r in rows); tn = sum((not r["pred"]) and (not r["label"]) for r in rows)
    p = tp / (tp + fp) if tp + fp else 0.0; rc = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": round(p, 3), "recall": round(rc, 3), "f1": round(2 * p * rc / (p + rc), 3) if p + rc else 0.0,
            "accuracy": round((tp + tn) / len(rows), 3), "n": len(rows)}


def run_policy(orch, ret, gen, items, policy, budget):
    rows, calls0 = [], dict(orch.verifier.calls)
    for it in items:
        ret.ev, gen.answer = it["evidence"], it["answer"]
        t = time.perf_counter()
        r = orch.run("benchmark", policy=policy, budget_s=budget, log=False)
        ms = (time.perf_counter() - t) * 1000
        types = [c["error_type"] for c in r["claims"]]
        pred = any(tp in FLAG_TYPES for tp in types) or not types
        main = next((tp for tp in types if tp in FLAG_TYPES), "SUPPORTED")
        rows.append({"id": it["id"], "type": it["type"], "label": it["label"], "pred": pred, "verdict": r["verdict"],
                     "pred_type": main, "type_ok": main == EXPECTED[it["type"]], "ms": round(ms, 1),
                     "plan": r["plan"], "degraded": r["budget_exceeded"],
                     "coverage": (sum(t != "UNCHECKED" for t in types) / len(types)) if types else 1.0})
    calls = {k: orch.verifier.calls[k] - calls0[k] for k in calls0}
    summ = {"policy": policy, "budget_s": budget, **prf(rows),
            "recall_by_type": {tp: round(sum(r["pred"] for r in rows if r["type"] == tp) /
                                         max(1, sum(r["type"] == tp for r in rows)), 3)
                               for tp in EXPECTED if tp != "FAITHFUL"},
            "false_alarm_rate": round(sum(r["pred"] for r in rows if not r["label"]) /
                                      max(1, sum(not r["label"] for r in rows)), 3),
            "type_accuracy_on_detected": round(sum(r["type_ok"] for r in rows if r["pred"] and r["label"]) /
                                               max(1, sum(r["pred"] and r["label"] for r in rows)), 3),
            "mean_ms": round(st.mean(r["ms"] for r in rows), 1),
            "p95_ms": round(sorted(r["ms"] for r in rows)[int(0.95 * (len(rows) - 1))], 1),
            "nli_calls_per_item": round((calls["tier1"] + calls["tier2"]) / len(rows), 2),
            "tier2_calls_per_item": round(calls["tier2"] / len(rows), 2),
            "degraded_share": round(sum(r["degraded"] for r in rows) / len(rows), 3),
            "claim_coverage": round(st.mean(r["coverage"] for r in rows), 3)}
    return summ, rows


def tables_md(summaries, sweep, n_items, counts):
    L = [f"## Table A - Detection by policy (MH-PolicyHalluBench, {n_items} items: {counts})", "",
         "| Policy | Precision | Recall | F1 | False-alarm | Type acc. | NLI calls/item | Mean ms | P95 ms |",
         "|---|---|---|---|---|---|---|---|---|"]
    for s in summaries:
        L.append(f"| {s['policy']} | {s['precision']} | {s['recall']} | {s['f1']} | {s['false_alarm_rate']} | "
                 f"{s['type_accuracy_on_detected']} | {s['nli_calls_per_item']} | {s['mean_ms']} | {s['p95_ms']} |")
    L += ["", "## Table B - Recall by hallucination type", "",
          "| Policy | " + " | ".join(summaries[0]["recall_by_type"]) + " |",
          "|---|" + "---|" * len(summaries[0]["recall_by_type"])]
    for s in summaries:
        L.append(f"| {s['policy']} | " + " | ".join(str(v) for v in s["recall_by_type"].values()) + " |")
    if sweep:
        L += ["", "## Table C - Latency-budget sweep (adaptive policy, verification stage only)", "",
              "| Budget (s) | F1 | Recall | Claims checked | Answers flagged Unverified | Mean ms |", "|---|---|---|---|---|---|"]
        for s in sweep:
            L.append(f"| {s['budget_s']} | {s['f1']} | {s['recall']} | {s['claim_coverage']} | {s['degraded_share']} | {s['mean_ms']} |")
    return "\n".join(L) + "\n"


def plots(summaries, sweep):
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib not installed -> skipping figures)"); return
    fig, ax = plt.subplots(figsize=(5, 3.5))
    for s in summaries:
        ax.scatter(s["mean_ms"], s["f1"], s=60); ax.annotate(s["policy"], (s["mean_ms"], s["f1"]),
                                                             textcoords="offset points", xytext=(5, 5))
    ax.set_xlabel("Mean verification latency per answer (ms)"); ax.set_ylabel("Detection F1")
    ax.set_title("Accuracy vs cost of verification policies"); fig.tight_layout()
    fig.savefig(RESULTS / "fig_policy_tradeoff.png", dpi=200); plt.close(fig)
    if sweep:
        fig, ax = plt.subplots(figsize=(5, 3.5)); b = [str(s["budget_s"]) for s in sweep]
        ax.plot(b, [s["f1"] for s in sweep], marker="o", label="F1")
        ax.plot(b, [s["claim_coverage"] for s in sweep], marker="s", label="Claims checked")
        ax.set_xlabel("Verification latency budget (s)"); ax.legend(); ax.set_title("Graceful degradation")
        fig.tight_layout(); fig.savefig(RESULTS / "fig_budget_sweep.png", dpi=200); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150, help="number of source chunks (items ~= 4-5 x n)")
    ap.add_argument("--seed", type=int, default=13); ap.add_argument("--mock-nli", action="store_true")
    ap.add_argument("--no-tier2", action="store_true"); ap.add_argument("--no-sweep", action="store_true")
    ap.add_argument("--policies", default="adaptive,always_full,heuristic_only")
    a = ap.parse_args()
    RESULTS.mkdir(exist_ok=True)
    items = build_items(a.n, a.seed)
    counts = {t: sum(i["type"] == t for i in items) for t in EXPECTED}
    print(f"Built {len(items)} items: {counts}")
    with open(RESULTS / "bench_items.jsonl", "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps({k: v for k, v in it.items() if k != "evidence"}, ensure_ascii=False) + "\n")
    ret, gen = FixedRetriever(), FixedGenerator()
    orch = Orchestrator(retriever=ret, generator=gen, mock_nli=a.mock_nli, load_tier2=not a.no_tier2)
    orch.verifier.warmup()
    summaries, all_rows = [], {}
    for pol in a.policies.split(","):
        s, rows = run_policy(orch, ret, gen, items, pol, None)
        summaries.append(s); all_rows[pol] = rows
        print(json.dumps(s, ensure_ascii=False))
    sweep = []
    if not a.no_sweep:
        for b in (0.5, 1.0, 2.0, 4.0, 8.0):   # must exceed RESERVE_PER_CLAIM_S to check anything
            s, _ = run_policy(orch, ret, gen, items, "adaptive", b); sweep.append(s)
            print(f"budget {b}s -> F1 {s['f1']}  degraded {s['degraded_share']}")
    json.dump({"n_items": len(items), "counts": counts, "policies": summaries, "budget_sweep": sweep,
               "nli_tier1": orch.verifier.tier1.name, "mock_nli": a.mock_nli},
              open(RESULTS / "bench_summary.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    with open(RESULTS / "bench_predictions.jsonl", "w", encoding="utf-8") as f:
        for pol, rows in all_rows.items():
            for r in rows:
                f.write(json.dumps({"policy": pol, **r}, ensure_ascii=False) + "\n")
    (RESULTS / "bench_tables.md").write_text(tables_md(summaries, sweep, len(items), counts), encoding="utf-8")
    plots(summaries, sweep)
    print(f"\nSaved -> {RESULTS}  (bench_tables.md is ready to paste into the paper)")


if __name__ == "__main__":
    main()
