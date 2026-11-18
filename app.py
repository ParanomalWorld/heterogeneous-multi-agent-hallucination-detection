"""Local web application (fully offline): citizen-facing policy assistant with live hallucination detection.

Run:   python app.py                      -> http://127.0.0.1:8000
       python app.py --mock               (UI test without LLM / NLI models)
       python app.py --model llama-3.1-8b --port 8000 --no-tier2
Everything (models, index, UI assets) is local. No CDN, no internet.
"""
import argparse, json, queue, re, threading, time, traceback
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from hallu_config import POLICIES, DEFAULT_POLICY, LATENCY_BUDGET_S, LOGS_DIR, NLI_TIER1, NLI_TIER2
from nov_config import DEFAULT_MODEL, TOP_K_EVIDENCE, ROOT

WEB = ROOT / "web"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)
app = FastAPI(title="Policy Hallucination Detector")
app.mount("/results", StaticFiles(directory=str(RESULTS)), name="results")
STATE = {"status": "starting", "error": None, "orch": None, "args": None, "loaded_at": None}
LOCK = threading.Lock()          # one CPU, one query at a time (queued requests wait)


def load(args):
    try:
        STATE["status"] = "loading models"
        from orchestrator import Orchestrator
        o = Orchestrator(args.model, mock_llm=args.mock, mock_nli=args.mock, load_tier2=not args.no_tier2)
        STATE["status"] = "warming up verifier"
        o.verifier.warmup()
        STATE.update(orch=o, status="ready", loaded_at=time.strftime("%H:%M:%S"))
    except Exception as e:  # shown in the UI instead of a silent crash
        STATE.update(status="error", error=f"{type(e).__name__}: {e}")
        traceback.print_exc()


# ------------------------------------------------------------------ demo-only hallucination injection
def make_injector(kind):
    if not kind or kind == "none":
        return None
    from hallu_bench import num_swap, negate, FABRICATIONS_MR
    import random
    rng = random.Random()
    def inj(answer, evidence):
        if kind == "number":
            out = num_swap(answer, rng)
        elif kind == "negation":
            out = negate(answer)
        else:
            latin = not re.search(r"[ऀ-ॿ]", answer)
            extra = ("Applicants must also hold a graduate degree to be eligible." if latin
                     else rng.choice(FABRICATIONS_MR))
            out = answer.rstrip() + " " + extra
        return out or answer.rstrip() + " " + rng.choice(FABRICATIONS_MR)
    return inj


class Ask(BaseModel):
    query: str
    policy: str = DEFAULT_POLICY
    budget_s: float | None = LATENCY_BUDGET_S
    k: int = TOP_K_EVIDENCE
    inject: str | None = None


@app.get("/")
def index():
    return FileResponse(WEB / "index.html")


@app.get("/api/health")
def health():
    o, a = STATE["orch"], STATE["args"]
    info = {"status": STATE["status"], "error": STATE["error"], "loaded_at": STATE["loaded_at"],
            "policies": POLICIES, "default_policy": DEFAULT_POLICY, "default_budget_s": LATENCY_BUDGET_S,
            "mock": bool(a and a.mock), "llm": a.model if a else None,
            "nli_tier1": NLI_TIER1, "nli_tier2": None if (a and a.no_tier2) else NLI_TIER2}
    if o:
        info.update(engine=o.engine, index_vectors=int(o.retriever.index.ntotal), busy=LOCK.locked())
    return info


@app.get("/api/examples")
def examples():
    p = ROOT / "eval_queries.json"
    qs = json.load(open(p, encoding="utf-8")) if p.exists() else []
    return [{"query": q["query"], "lang": q.get("query_lang"), "in_scope": q.get("in_scope", True)} for q in qs]


@app.post("/api/ask")
def ask(req: Ask):
    if STATE["status"] != "ready":
        raise HTTPException(503, f"Models not ready: {STATE['status']} {STATE['error'] or ''}")
    if req.policy not in POLICIES:
        raise HTTPException(400, "unknown policy")
    if not req.query.strip():
        raise HTTPException(400, "empty query")
    q: queue.Queue = queue.Queue()

    def work():
        waited = LOCK.locked()
        if waited:
            q.put({"type": "status", "text": "Another query is running - waiting in queue..."})
        with LOCK:
            try:
                STATE["orch"].run(req.query.strip(), policy=req.policy, budget_s=req.budget_s or None,
                                  k=max(1, min(req.k, 8)), emit=q.put, inject=make_injector(req.inject))
            except Exception as e:
                traceback.print_exc()
                q.put({"type": "error", "text": f"{type(e).__name__}: {e}"})
            finally:
                q.put(None)

    threading.Thread(target=work, daemon=True).start()

    def stream():
        while (ev := q.get()) is not None:
            yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


def _runs():
    p = LOGS_DIR / "orchestrator_runs.jsonl"
    if not p.exists():
        return []
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


@app.get("/api/history")
def history(limit: int = 50):
    rows = _runs()[-limit:][::-1]
    return [{k: r.get(k) for k in ("id", "time", "query", "verdict", "risk", "faithfulness", "policy", "plan",
                                   "abstained", "budget_exceeded", "model")} | {"total_s": round(r["timing_ms"]["total"] / 1000, 2)}
            for r in rows]


@app.get("/api/report/{rid}")
def report(rid: str):
    for r in _runs()[::-1]:
        if r["id"] == rid:
            return JSONResponse(r, headers={"Content-Disposition": f"attachment; filename=audit_{rid}.json"})
    raise HTTPException(404, "not found")


@app.get("/api/stats")
def stats():
    runs = _runs()
    out = {"live": None, "bench": None, "e2e": None, "figures": sorted(p.name for p in RESULTS.glob("*.png"))}
    if runs:
        from collections import Counter
        import statistics as st
        out["live"] = {"n": len(runs), "verdicts": dict(Counter(r["verdict"] for r in runs)),
                       "mean_total_s": round(st.mean(r["timing_ms"]["total"] for r in runs) / 1000, 2),
                       "mean_stage_ms": {k: round(st.mean(r["timing_ms"].get(k, 0) for r in runs), 1)
                                         for k in ("retrieval", "generation", "verification", "critic")},
                       "error_types": dict(sum((Counter(r["error_counts"]) for r in runs), Counter()))}
    for key, name in (("bench", "bench_summary.json"), ("e2e", "e2e_summary.json")):
        if (RESULTS / name).exists():
            out[key] = json.load(open(RESULTS / name, encoding="utf-8"))
    if (RESULTS / "bench_tables.md").exists():
        out["bench_tables_md"] = (RESULTS / "bench_tables.md").read_text(encoding="utf-8")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL); ap.add_argument("--mock", action="store_true")
    ap.add_argument("--no-tier2", action="store_true"); ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    a = ap.parse_args()
    STATE["args"] = a
    threading.Thread(target=load, args=(a,), daemon=True).start()   # UI opens immediately, models load behind
    print(f"\n  Open  http://{a.host}:{a.port}   (models load in the background; status shown in the page)\n")
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")
