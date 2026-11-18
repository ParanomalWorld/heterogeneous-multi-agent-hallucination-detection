"""Orchestrator Agent (Jan milestone) + Latency Budget Monitor + full 5-agent pipeline (Feb integration).

Graph (LangGraph StateGraph if installed, otherwise an identical built-in runner):

    retrieve --(top score < EARLY_EXIT_T)--> critic            (early abstain, LLM skipped)
        |
    generate --(abstained / policy=no_verify)--> critic
        |
    precheck   Tier-0 on every claim -> response risk -> verification PLAN (fast / deep / full / heuristic)
        |
    verify     claims checked riskiest-first; Tier-1 NLI, escalate uncertain claims to Tier-2;
        |      before every check the Latency Budget Monitor is asked -> claims left when time runs out
        |      are marked UNCHECKED (graceful degradation: user still gets the answer, flagged "Unverified")
    critic     error type per claim, verdict, risk  -> streamed to the user with the decision trace

Usage:
  python orchestrator.py "अंगणवाडी सेविकांना मानधन किती मिळते?"
  python orchestrator.py "..." --policy always_full --budget 30
  python orchestrator.py "..." --mock          (no LLM, no NLI: plumbing test)
"""
import json, time, uuid
from typing import Any, TypedDict

from hallu_config import (LATENCY_BUDGET_S, RESERVE_PER_CLAIM_S, WEAK_RETRIEVAL_T, EARLY_EXIT_T, LOW_RISK_T,
                          DEFAULT_POLICY, POLICIES, LOGS_DIR)
from nov_config import DEFAULT_MODEL, TOP_K_EVIDENCE, EVIDENCE_LANG, NOT_FOUND_TOKEN
from text_utils import split_claims, evidence_sentences, numbers
from verifier_agent import VerifierAgent
from critic_agent import CriticAgent


class LatencyBudgetMonitor:
    def __init__(self, budget_s):
        self.budget = float(budget_s) if budget_s else float("inf")
        self.t0 = time.perf_counter()
        self.stages = {}

    def elapsed(self):
        return time.perf_counter() - self.t0

    def remaining(self):
        return self.budget - self.elapsed()

    def can_afford(self, cost_s):
        return self.remaining() >= cost_s

    def mark(self, stage, start):
        self.stages[stage] = round((time.perf_counter() - start) * 1000, 1)

    def snapshot(self):
        b = None if self.budget == float("inf") else self.budget
        return {"elapsed_s": round(self.elapsed(), 2), "budget_s": b,
                "remaining_s": None if b is None else round(self.remaining(), 2)}


class PState(TypedDict, total=False):
    query: str
    policy: str
    k: int
    lang: Any
    retrieval: Any
    generation: Any
    claims: Any
    plan: Any
    response_risk: float
    budget_exceeded: bool
    assessment: Any
    route: str


class Orchestrator:
    def __init__(self, model_key=DEFAULT_MODEL, mock_llm=False, mock_nli=False, load_tier2=True,
                 use_langgraph=True, retriever=None, generator=None):
        from retriever_agent import RetrieverAgent
        from generator_agent import GeneratorAgent
        self.retriever = retriever or RetrieverAgent()
        self.generator = generator or GeneratorAgent(model_key=model_key, mock=mock_llm)
        self.verifier = VerifierAgent(mock=mock_nli, load_tier2=load_tier2)
        self.critic = CriticAgent()
        self.emit = lambda ev: None
        self.monitor = None
        self.trace = []
        self.graph = self._build_langgraph() if use_langgraph else None
        self.engine = "langgraph" if self.graph else "builtin"

    # ------------------------------------------------------------------ helpers
    def _log(self, agent, decision, why, **extra):
        item = {"t_s": round(self.monitor.elapsed(), 3), "agent": agent, "decision": decision, "why": why, **extra}
        self.trace.append(item)
        self.emit({"type": "trace", **item})

    # ------------------------------------------------------------------ nodes
    def n_retrieve(self, s):
        st = time.perf_counter(); self.emit({"type": "agent", "agent": "retriever", "status": "running"})
        r = self.retriever.retrieve(s["query"], k=s["k"], lang=s["lang"])
        self.monitor.mark("retrieval", st)
        self.emit({"type": "evidence", "evidence": r["evidence"], "top_score": r["top_score"]})
        self.emit({"type": "agent", "agent": "retriever", "status": "done", "ms": self.monitor.stages["retrieval"]})
        route = "early_abstain" if r["top_score"] < EARLY_EXIT_T else "generate"
        self._log("orchestrator", route, f"retrieval top score {r['top_score']:.3f} "
                  f"({'<' if route == 'early_abstain' else '>='} early-exit {EARLY_EXIT_T})")
        return {"retrieval": r, "route": route}

    def n_generate(self, s):
        st = time.perf_counter(); self.emit({"type": "agent", "agent": "generator", "status": "running"})
        g = self.generator.generate(s["query"], s["retrieval"]["evidence"],
                                    on_token=lambda t: self.emit({"type": "token", "text": t}))
        self.monitor.mark("generation", st)
        self.emit({"type": "agent", "agent": "generator", "status": "done", "ms": self.monitor.stages["generation"],
                   "confidence": g["confidence"]})
        if g["abstained"]:
            route, why = "critic", "generator abstained (NOT_FOUND) -> nothing to verify"
        elif s["policy"] == "no_verify":
            route, why = "critic", "policy no_verify (baseline) -> skip verification"
        else:
            route, why = "precheck", f"answer produced, confidence={g['confidence']}"
        self._log("orchestrator", route, why, budget=self.monitor.snapshot())
        return {"generation": g, "route": route}

    def n_precheck(self, s):
        st = time.perf_counter(); self.emit({"type": "agent", "agent": "verifier", "status": "running"})
        ev = s["generation"]["used_evidence"] or s["retrieval"]["evidence"]
        self._ev_sents = evidence_sentences(ev)
        all_nums = set().union(*(numbers(e["text"]) for e in ev)) if ev else set()
        qn = numbers(s["query"])
        claims = []
        for c in split_claims(s["generation"]["answer"]):
            t0 = self.verifier.tier0(c, self._ev_sents, all_nums, qn)
            claims.append({**c, "tier0": t0, "tier": 0, "label": self.verifier.tier0_label(t0),
                           "status": "pending", "probs": None})
        g, r = s["generation"], s["retrieval"]
        max_claim = max((c["tier0"]["risk"] for c in claims), default=1.0)
        conf_risk = 0.5 if g["confidence"] is None else (1 - g["confidence"])
        risk = round(min(1.0, 0.55 * max_claim + 0.25 * conf_risk
                         + (0.2 if r["top_score"] < WEAK_RETRIEVAL_T else 0.0)), 3)
        pol = s["policy"]
        if pol == "heuristic_only":
            plan = "heuristic"
        elif pol == "always_full":
            plan = "full"
        else:
            plan = "fast" if risk < LOW_RISK_T else "deep"
        self.monitor.mark("precheck", st)
        self.emit({"type": "claims", "claims": _public_claims(claims)})
        self._log("orchestrator", f"plan={plan}", f"response risk {risk} (max claim risk {max_claim}, "
                  f"generator risk {conf_risk:.2f}); low-risk threshold {LOW_RISK_T}; policy {pol}",
                  budget=self.monitor.snapshot())
        return {"claims": claims, "plan": plan, "response_risk": risk}

    def n_verify(self, s):
        st = time.perf_counter()
        claims, plan, exceeded = s["claims"], s["plan"], False
        order = sorted(claims, key=lambda c: -c["tier0"]["risk"])   # riskiest first: best use of a tight budget
        for c in order:
            if plan == "heuristic":
                c["status"] = "checked"; continue
            if plan == "fast" and not (c["tier0"]["claim_numbers"] or c["tier0"]["signal"] != "lexical_support"):
                c["status"] = "checked"                                    # tier-0 is enough for this claim
                self._log("verifier", f"claim {c['id']}: tier-0 only", "low-risk plan, lexically supported, no numbers")
                self.emit({"type": "claim", "claim": _public_claims([c])[0]}); continue
            if not self.monitor.can_afford(RESERVE_PER_CLAIM_S):
                exceeded = True
                if c["tier0"]["missing_numbers"]:      # tier-0 evidence is free: still report a fabricated number
                    c["status"] = "checked"
                    self._log("orchestrator", f"claim {c['id']}: tier-0 only", "budget exhausted, but tier-0 found "
                              "a number absent from all evidence", budget=self.monitor.snapshot())
                    self.emit({"type": "claim", "claim": _public_claims([c])[0]}); continue
                c["status"] = "unchecked"
                self._log("orchestrator", f"claim {c['id']}: UNCHECKED", "latency budget exhausted -> graceful degradation",
                          budget=self.monitor.snapshot())
                self.emit({"type": "claim", "claim": _public_claims([c])[0]}); continue
            r = self.verifier.tier1_check(c, c["tier0"])
            res, why = r, f"tier-1 {r['label']} p={r['probs']}"
            want_t2 = plan == "full" or (r["uncertain"] and plan == "deep")
            if want_t2 and self.monitor.can_afford(RESERVE_PER_CLAIM_S * 3):
                r2 = self.verifier.tier2_check(c, c["tier0"], self._ev_sents)
                res, why = r2, why + f" -> escalated: tier-2 {r2['label']} ({r2['model'].split('/')[-1]})"
            elif want_t2:
                why += " -> tier-2 wanted but no budget left"
            c.update({"tier": res["tier"], "probs": res["probs"], "nli_model": res["model"],
                      "label": self.verifier.final_label(c["tier0"], res), "status": "checked", "nli_ms": res["ms"]})
            self._log("verifier", f"claim {c['id']}: {c['label']}", why)
            self.emit({"type": "claim", "claim": _public_claims([c])[0]})
        self.monitor.mark("verification", st)
        self.emit({"type": "agent", "agent": "verifier", "status": "done", "ms": self.monitor.stages["verification"]})
        return {"claims": claims, "budget_exceeded": exceeded}

    def n_critic(self, s):
        st = time.perf_counter(); self.emit({"type": "agent", "agent": "critic", "status": "running"})
        g = s.get("generation") or {"abstained": True, "confidence": None, "answer": NOT_FOUND_TOKEN}
        a = self.critic.assess(s.get("claims") or [], g["abstained"], s.get("budget_exceeded", False),
                               s["policy"], s["retrieval"]["top_score"], g.get("confidence"))
        self.monitor.mark("critic", st)
        self.emit({"type": "agent", "agent": "critic", "status": "done", "ms": self.monitor.stages["critic"]})
        self._log("critic", a["verdict"], f"risk {a['risk']}, faithfulness {a['faithfulness']}, "
                  f"errors {a['error_counts']}")
        return {"assessment": a}

    # ------------------------------------------------------------------ graph
    @staticmethod
    def _r(s):
        return s["route"]

    def _build_langgraph(self):
        try:
            from langgraph.graph import StateGraph, END
        except ImportError:
            return None
        g = StateGraph(PState)
        for name in ("retrieve", "generate", "precheck", "verify", "critic"):
            g.add_node(name, (lambda n: lambda s: getattr(self, f"n_{n}")(s))(name))   # late binding
        g.set_entry_point("retrieve")
        g.add_conditional_edges("retrieve", self._r, {"generate": "generate", "early_abstain": "critic"})
        g.add_conditional_edges("generate", self._r, {"precheck": "precheck", "critic": "critic"})
        g.add_edge("precheck", "verify"); g.add_edge("verify", "critic"); g.add_edge("critic", END)
        return g.compile()

    def _run_builtin(self, s):
        s.update(self.n_retrieve(s))
        if s["route"] == "generate":
            s.update(self.n_generate(s))
            if s["route"] == "precheck":
                s.update(self.n_precheck(s)); s.update(self.n_verify(s))
        s.update(self.n_critic(s))
        return s

    # ------------------------------------------------------------------ public
    def run(self, query, policy=DEFAULT_POLICY, budget_s=LATENCY_BUDGET_S, k=TOP_K_EVIDENCE, lang=EVIDENCE_LANG,
            emit=None, log=True, inject=None):
        """inject: optional fn(answer, evidence)->answer used ONLY by the demo/benchmark to plant a hallucination."""
        assert policy in POLICIES, policy
        self.emit, self.trace = (emit or (lambda ev: None)), []
        self.monitor = LatencyBudgetMonitor(budget_s)
        self._inject = inject
        self._log("orchestrator", "start", f"policy={policy}, budget={budget_s}s, engine={self.engine}")
        state = {"query": query, "policy": policy, "k": k, "lang": lang}
        if inject:      # wrap generation so the planted error flows through the normal graph
            orig = self.n_generate
            def gen_with_injection(s):
                out = orig(s)
                if not out["generation"]["abstained"]:
                    out["generation"]["answer"] = inject(out["generation"]["answer"], s["retrieval"]["evidence"])
                    self.emit({"type": "answer_replaced", "answer": out["generation"]["answer"]})
                    self._log("demo", "hallucination injected", "test mode: answer perturbed before verification")
                return out
            self.n_generate = gen_with_injection
        try:
            final = self.graph.invoke(state) if self.graph else self._run_builtin(state)
        finally:
            if inject:
                self.n_generate = orig
        res = self._result(final, policy)
        if log:
            LOGS_DIR.mkdir(exist_ok=True)
            with open(LOGS_DIR / "orchestrator_runs.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(res, ensure_ascii=False) + "\n")
        self.emit({"type": "final", "result": res})
        return res

    def _result(self, s, policy):
        g, r, a = s.get("generation") or {}, s["retrieval"], s["assessment"]
        return {"id": uuid.uuid4().hex[:10], "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "query": s["query"], "policy": policy, "engine": self.engine,
                "answer": g.get("answer", NOT_FOUND_TOKEN), "abstained": g.get("abstained", True),
                "generator_confidence": g.get("confidence"), "model": g.get("model"),
                "retrieval_top_score": r["top_score"], "plan": s.get("plan"), "response_risk": s.get("response_risk"),
                **a, "claims": _public_claims(s.get("claims") or []),
                "sources": [{k: e.get(k) for k in ("rank", "chunk_id", "doc_id", "department", "gr_no", "pages", "score")}
                            for e in r["evidence"]],
                "timing_ms": {**self.monitor.stages, "total": round(self.monitor.elapsed() * 1000, 1)},
                "budget": self.monitor.snapshot(), "budget_exceeded": s.get("budget_exceeded", False),
                "trace": self.trace}


def _public_claims(claims):
    keep = ("id", "text", "cites", "status", "label", "tier", "probs", "error_type", "reason", "nli_model")
    out = []
    for c in claims:
        d = {k: c.get(k) for k in keep}
        t0 = c["tier0"]
        d["tier0"] = {k: t0[k] for k in ("signal", "lexical", "missing_numbers", "risk", "cited_ok")}
        d["premise"] = [{"ev": p["ev"], "text": p["text"]} for p in t0["premise"]]
        out.append(d)
    return out


def show(res):
    print("\n" + "=" * 78)
    print("Q:", res["query"]); print("A:", res["answer"])
    print(f"\nVERDICT: {res['verdict']} ({res['verdict_mr']})  risk={res['risk']}  faithfulness={res['faithfulness']}"
          f"  plan={res['plan']}  depth={res['verification_depth']}")
    for c in res["claims"]:
        print(f"  [{c['error_type']:<20}] tier{c['tier']} {c['text'][:90]}")
        if c["error_type"] != "SUPPORTED":
            print(f"      -> {c['reason']}")
    print("timing_ms:", res["timing_ms"], "| budget:", res["budget"])
    print("trace:"); [print(f"  {t['t_s']:>7.2f}s {t['agent']:<12} {t['decision']:<28} {t['why'][:80]}") for t in res["trace"]]
    print("=" * 78)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("query", nargs="?"); ap.add_argument("--chat", action="store_true")
    ap.add_argument("--policy", default=DEFAULT_POLICY, choices=POLICIES)
    ap.add_argument("--budget", type=float, default=LATENCY_BUDGET_S)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--mock", action="store_true", help="mock LLM + mock NLI")
    ap.add_argument("--mock-nli", action="store_true"); ap.add_argument("--no-tier2", action="store_true")
    a = ap.parse_args()
    o = Orchestrator(a.model, mock_llm=a.mock, mock_nli=a.mock or a.mock_nli, load_tier2=not a.no_tier2)
    if a.chat:
        while (q := input("\nप्रश्न / Question (exit): ").strip()) and q.lower() != "exit":
            show(o.run(q, a.policy, a.budget))
    elif a.query:
        show(o.run(a.query, a.policy, a.budget))
    else:
        ap.error("give a query or --chat")
