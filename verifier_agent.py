"""Verifier Agent (Dec milestone): sentence-level faithfulness scoring of generated claims against retrieved
policy evidence, as a HETEROGENEOUS cascade of verifiers with very different costs:

  Tier-0  heuristic : numeric grounding + lexical support. No model, ~1 ms.
  Tier-1  small NLI : multilingual mDeBERTa-v3-base (int8), best premise only. ~0.1-0.4 s on CPU.
  Tier-2  deep NLI  : several premise windows; DeBERTa-v3-large-MNLI for English/English pairs,
                      else the multilingual model over all windows (max-pooled). ~1-3 s on CPU.

The Verifier only EXECUTES checks; WHICH tier runs for WHICH claim is decided by the Orchestrator
(confidence, risk and remaining latency budget). Detection only: the answer is never rewritten.

CLI test:  python verifier_agent.py --mock
"""
import time, warnings
from hallu_config import (NLI_TIER1, NLI_TIER2, NLI_DIR, NLI_QUANTIZE_INT8, NLI_MAX_TOKENS, PREMISE_SENTENCES,
                          TIER2_ONLY_LATIN, ENTAIL_T, CONTRA_T, UNCERTAIN_MARGIN, LEXICAL_SUPPORT_T)
from text_utils import numbers, select_premise, lexical_support

LABELS = ("entailment", "neutral", "contradiction")


def is_latin(t: str) -> bool:
    dev = sum("ऀ" <= ch <= "ॿ" for ch in t)
    letters = sum(ch.isalpha() for ch in t) or 1
    return dev / letters < 0.1


class NLIModel:
    """HuggingFace sequence-classification NLI model on CPU, optional int8 dynamic quantisation."""
    def __init__(self, name, quantize=NLI_QUANTIZE_INT8):
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        local = NLI_DIR / name.split("/")[-1]
        src = str(local) if local.exists() else name          # prefer the offline copy from download_nli.py
        self.name = name
        self.tok = AutoTokenizer.from_pretrained(src)
        # checkpoints may be stored in float16; CPU inference + int8 quantisation need float32
        model = AutoModelForSequenceClassification.from_pretrained(src).float().eval()
        self.torch, self.model, self.quantized = torch, model, False
        id2label = {int(k): v.lower() for k, v in model.config.id2label.items()}
        self.idx = {lab: next(i for i, v in id2label.items() if v.startswith(lab[:7])) for lab in LABELS}
        torch.set_num_threads(max(1, (__import__("os").cpu_count() or 4) - 1))
        if quantize:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    qm = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
                    self.model = qm
                    self.predict([("test premise", "test hypothesis")])     # fail fast here, not mid-query
                self.quantized = True
            except Exception as e:      # some torch/CPU builds lack int8 kernels -> keep float32 (slower, same labels)
                print(f"[verifier] int8 quantisation unavailable ({type(e).__name__}); using float32")
                self.model = model

    def predict(self, pairs):
        """pairs: [(premise, hypothesis)] -> [{'entailment':p,'neutral':p,'contradiction':p}]"""
        with self.torch.inference_mode():
            enc = self.tok([p for p, _ in pairs], [h for _, h in pairs], truncation="only_first",
                           max_length=NLI_MAX_TOKENS, padding=True, return_tensors="pt")
            probs = self.model(**enc).logits.softmax(-1).tolist()
        return [{lab: round(pr[self.idx[lab]], 4) for lab in LABELS} for pr in probs]


class MockNLI:
    """Model-free stand-in (for tests / laptops without the NLI download). Derived from tier-0 signals."""
    name = "mock-nli"
    def predict(self, pairs):
        out = []
        for prem, hyp in pairs:
            hn, pn = numbers(hyp), numbers(prem)
            lex = lexical_support(hyp, prem)
            if hn and not hn <= pn:
                out.append({"entailment": 0.1, "neutral": 0.25, "contradiction": 0.65})
            elif lex >= LEXICAL_SUPPORT_T:
                out.append({"entailment": 0.8, "neutral": 0.15, "contradiction": 0.05})
            else:
                out.append({"entailment": 0.2, "neutral": 0.7, "contradiction": 0.1})
        return out


def decide(p):
    """NLI probabilities -> claim label + whether the decision is uncertain (used for escalation)."""
    ranked = sorted(p.items(), key=lambda kv: -kv[1])
    uncertain = ranked[0][1] - ranked[1][1] < UNCERTAIN_MARGIN
    if p["entailment"] >= ENTAIL_T:
        lab = "SUPPORTED"
    elif p["contradiction"] >= CONTRA_T:
        lab = "CONTRADICTED"
    elif ranked[0][0] == "neutral":
        lab = "NOT_SUPPORTED"
    else:
        lab, uncertain = "UNCERTAIN", True
    return lab, uncertain


class VerifierAgent:
    def __init__(self, mock=False, load_tier2=True):
        self.mock, self.load_tier2 = mock, load_tier2
        self._t1 = self._t2 = None
        self.calls = {"tier1": 0, "tier2": 0}     # cost accounting for the benchmark

    # lazy loading keeps RAM free until a tier is actually needed (resource-constrained design)
    @property
    def tier1(self):
        if self._t1 is None:
            self._t1 = MockNLI() if self.mock else NLIModel(NLI_TIER1)
        return self._t1

    @property
    def tier2_large(self):
        if self._t2 is None:
            self._t2 = MockNLI() if (self.mock or not self.load_tier2) else NLIModel(NLI_TIER2)
        return self._t2

    def warmup(self):
        self.tier1.predict([("warm up", "warm up")])

    # ---------------- Tier-0 -------------------------------------------------------------------
    def tier0(self, claim, ev_sents, all_evidence_numbers, query_numbers=frozenset()):
        t0 = time.perf_counter()
        premise, best_lex = select_premise(claim["text"], ev_sents, PREMISE_SENTENCES, prefer_ev=claim["cites"])
        cn = numbers(claim["text"]) - set(query_numbers)
        missing = sorted(n for n in cn if n not in all_evidence_numbers)
        lex = max((lexical_support(claim["text"], s["text"]) for s in premise), default=0.0)
        if missing:
            signal = "numeric_mismatch"
        elif lex >= LEXICAL_SUPPORT_T:
            signal = "lexical_support"
        else:
            signal = "weak_support"
        cited_ok = (not claim["cites"]) or any(s["ev"] in claim["cites"] for s in premise[:1])
        # pre-verification risk of this claim (0 = safe .. 1 = very risky); used by the Orchestrator to order checks
        risk = min(1.0, (0.6 if missing else 0.0) + 0.4 * (1 - lex) + (0.1 if cn else 0.0)
                   + (0.1 if not claim["cites"] else 0.0) + (0.05 if not cited_ok else 0.0))
        return {"premise": premise, "lexical": round(lex, 3), "claim_numbers": sorted(cn),
                "missing_numbers": missing, "signal": signal, "cited_ok": cited_ok,
                "has_citation": bool(claim["cites"]), "risk": round(risk, 3),
                "ms": round((time.perf_counter() - t0) * 1000, 2)}

    @staticmethod
    def tier0_label(t0):
        if t0["missing_numbers"]:
            return "NUMERIC_MISMATCH"
        return "LIKELY_SUPPORTED" if t0["signal"] == "lexical_support" else "UNCERTAIN"

    # ---------------- Tier-1 / Tier-2 ----------------------------------------------------------
    def tier1_check(self, claim, t0):
        t = time.perf_counter(); self.calls["tier1"] += 1
        prem = " ".join(s["text"] for s in t0["premise"])
        p = self.tier1.predict([(prem, claim["text"])])[0]
        lab, unc = decide(p)
        return {"tier": 1, "model": self.tier1.name, "probs": p, "label": lab, "uncertain": unc,
                "ms": round((time.perf_counter() - t) * 1000, 1)}

    def tier2_check(self, claim, t0, ev_sents):
        """Deep check: several premise windows, max-pooled. Large English model when both sides are English."""
        t = time.perf_counter(); self.calls["tier2"] += 1
        windows, _ = select_premise(claim["text"], ev_sents, 6, prefer_ev=claim["cites"])
        prem_all = " ".join(s["text"] for s in t0["premise"])
        premises = [prem_all] + [s["text"] for s in windows]
        use_large = (self.mock or self.load_tier2) and ((not TIER2_ONLY_LATIN) or
                                                         (is_latin(claim["text"]) and is_latin(prem_all)))
        model = self.tier2_large if use_large else self.tier1
        ps = model.predict([(pr, claim["text"]) for pr in premises])
        # supported if ANY window entails; contradicted only if the best-matching windows contradict
        e = max(p["entailment"] for p in ps)
        c = max(p["contradiction"] for p in ps[:3])
        n = max(0.0, 1 - e - c)
        p = {"entailment": round(e, 4), "neutral": round(n, 4), "contradiction": round(c, 4)}
        lab, unc = decide(p)
        return {"tier": 2, "model": model.name, "probs": p, "label": lab, "uncertain": unc,
                "windows": len(premises), "ms": round((time.perf_counter() - t) * 1000, 1)}

    @staticmethod
    def final_label(t0, nli):
        """Combine NLI with the numeric guard: a number that occurs in NO evidence can't be 'supported'."""
        if nli is None:
            return VerifierAgent.tier0_label(t0)
        lab = nli["label"]
        if t0["missing_numbers"] and lab == "SUPPORTED":
            return "UNCERTAIN"
        return lab


if __name__ == "__main__":
    import argparse, json
    from text_utils import split_claims, evidence_sentences
    ap = argparse.ArgumentParser(); ap.add_argument("--mock", action="store_true"); a = ap.parse_args()
    ev = [{"chunk_id": "c1", "text": "अंगणवाडी सेविकांच्या मानधनात वाढ करून ते दरमहा रु. 10,000 करण्यात आले आहे. "
                                     "मदतनीस यांना दरमहा रु. 5,500 मानधन देण्यात येईल."}]
    ans = "अंगणवाडी सेविकांना दरमहा रु. 15,000 मानधन मिळते [1]. मदतनीस यांना दरमहा रु. 5,500 मानधन मिळेल [1]."
    v, sents = VerifierAgent(mock=a.mock), evidence_sentences(ev)
    nums = set().union(*(numbers(e["text"]) for e in ev))
    for c in split_claims(ans):
        t0 = v.tier0(c, sents, nums); r1 = v.tier1_check(c, t0)
        print(json.dumps({"claim": c["text"], "tier0": {k: t0[k] for k in ("signal", "missing_numbers", "risk")},
                          "tier1": r1, "final": v.final_label(t0, r1)}, ensure_ascii=False, indent=1))
