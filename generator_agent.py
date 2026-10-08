"""Generator Agent: evidence-grounded answer generation with a locally hosted, quantised Llama (llama.cpp).

Design (maps to synopsis): takes the Retriever's evidence, produces a candidate answer and a confidence score.
The answer must cite evidence numbers [1],[2].. and say NOT_FOUND when the evidence does not contain the answer.
Detection only: this agent never rewrites/corrects; verification happens in the Verifier Agent (Dec).
"""
import math, re, time
from nov_config import (MODEL_REGISTRY, DEFAULT_MODEL, MODELS_DIR, N_CTX, MAX_NEW_TOKENS,
                        TEMPERATURE, NOT_FOUND_TOKEN)

SYSTEM_PROMPT = (
    "You are a careful assistant for Maharashtra Government scheme information.\n"
    "Rules:\n"
    "1. Answer ONLY using the numbered EVIDENCE below. Do not use outside knowledge.\n"
    "2. After each fact write the evidence number in brackets, like [1] or [2].\n"
    "3. Copy numbers, amounts, dates and names exactly as written in the evidence.\n"
    f"4. If the evidence does not contain the answer, reply with exactly: {NOT_FOUND_TOKEN}\n"
    "5. Reply in the same language as the question (Marathi question -> Marathi answer, English -> English).\n"
    "6. Keep the answer short (2-5 sentences)."
)


def build_user_prompt(query, evidence):
    ev = "\n\n".join(f"[{i}] {e['text']}" for i, e in enumerate(evidence, 1))
    return f"EVIDENCE:\n{ev}\n\nQUESTION: {query}\n\nANSWER:"


class GeneratorAgent:
    def __init__(self, model_key=DEFAULT_MODEL, mock=False, n_threads=None, n_ctx=N_CTX):
        self.mock, self.model_key, self.n_ctx = mock, model_key, n_ctx
        self.llm = None
        if mock:
            return
        from llama_cpp import Llama
        path = MODELS_DIR / MODEL_REGISTRY[model_key]["file"]
        if not path.exists():
            raise FileNotFoundError(f"{path} not found. Run: python download_model.py {model_key}")
        self.llm = Llama(model_path=str(path), n_ctx=n_ctx, n_threads=n_threads,
                         n_gpu_layers=0, logits_all=False, verbose=False)

    # ---- helpers -------------------------------------------------------
    def _count(self, text):
        if self.mock:
            return len(text) // 2
        return len(self.llm.tokenize(text.encode("utf-8"), add_bos=False))

    def fit_evidence(self, query, evidence, max_new):
        """Drop lowest-ranked chunks until the prompt fits the context window (Marathi uses many tokens)."""
        budget = self.n_ctx - max_new - 200
        ev = list(evidence)
        while ev and self._count(SYSTEM_PROMPT + build_user_prompt(query, ev)) > budget:
            ev.pop()
        return ev

    @staticmethod
    def _confidence(logprobs_content):
        """Mean token probability of the answer (0-1). Simple, cheap uncertainty signal for the Orchestrator."""
        lps = [t["logprob"] for t in (logprobs_content or []) if "logprob" in t]
        return round(math.exp(sum(lps) / len(lps)), 4) if lps else None

    # ---- main ----------------------------------------------------------
    def generate(self, query, evidence, max_new_tokens=MAX_NEW_TOKENS):
        t0 = time.perf_counter()
        used = self.fit_evidence(query, evidence, max_new_tokens)
        if self.mock:   # test mode only: no LLM, returns the first evidence snippet
            text = (f"[mock] {used[0]['text'][:200]} [1]" if used else NOT_FOUND_TOKEN)
            conf, n_tok = 0.5, 0
        else:
            out = self.llm.create_chat_completion(
                messages=[{"role": "system", "content": SYSTEM_PROMPT},
                          {"role": "user", "content": build_user_prompt(query, used)}],
                max_tokens=max_new_tokens, temperature=TEMPERATURE)
            ch = out["choices"][0]
            text = ch["message"]["content"].strip()
            conf = None   # token-probability confidence: added later, memory-heavy on 8 GB
            n_tok = out["usage"]["completion_tokens"]
        abstained = NOT_FOUND_TOKEN in text
        cited = sorted({int(n) for n in re.findall(r"\[(\d+)\]", text) if 1 <= int(n) <= len(used)})
        return {"answer": text, "abstained": abstained, "confidence": conf,
                "cited_chunk_ids": [used[n - 1]["chunk_id"] for n in cited],
                "evidence_used": len(used), "completion_tokens": n_tok,
                "model": "mock" if self.mock else self.model_key,
                "latency_ms": round((time.perf_counter() - t0) * 1000, 1)}
