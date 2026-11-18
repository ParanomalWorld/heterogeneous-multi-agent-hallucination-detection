"""Phase-2 settings (Dec-Feb milestones): Verifier, Critic, Orchestrator, Web app.
Change thresholds HERE, never inside the agents, so every experiment is reproducible.
Every value below is reported in the thesis / paper "Experimental setup" table.
"""
import os
from pathlib import Path

ROOT = Path(__file__).parent
LOGS_DIR = ROOT / "logs"
NLI_DIR = ROOT / "models" / "nli"

# ---------------------------------------------------------------------------------------------
# Heterogeneous verifier cascade (cheap -> expensive). Only the tiers that are needed are run.
#   Tier-0  heuristic   : numeric grounding + lexical support          (~1 ms / claim, no model)
#   Tier-1  small NLI   : multilingual mDeBERTa-v3-base (XNLI, 100 langs incl. Marathi script) (~0.1-0.4 s / claim on CPU)
#   Tier-2  large NLI   : DeBERTa-v3-large MNLI (synopsis model)        (~1-2 s / claim on CPU, English claims)
# ---------------------------------------------------------------------------------------------
NLI_TIER1 = os.environ.get("NLI_TIER1", "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7")
NLI_TIER2 = os.environ.get("NLI_TIER2", "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli")
NLI_QUANTIZE_INT8 = os.environ.get("HALLU_NLI_INT8", "0") == "1"   # int8 = faster/smaller but can hurt DeBERTa accuracy;
                                # check with `python nli_selftest.py` before switching it on (ablation in the paper)
NLI_MAX_TOKENS = 512
PREMISE_SENTENCES = 3           # evidence sentences given to NLI as premise (best-matching ones)
TIER2_ONLY_LATIN = True         # large English NLI is only escalated to for English (Latin-script) claims

# Decision thresholds
ENTAIL_T = 0.60                 # P(entail)  >= this -> SUPPORTED
CONTRA_T = 0.60                 # P(contra)  >= this -> CONTRADICTED
UNCERTAIN_MARGIN = 0.25         # |top1 - top2| below this -> tier-1 is "uncertain" -> escalate to tier-2
LEXICAL_SUPPORT_T = 0.55        # tier-0: share of claim tokens found in best evidence sentence

# ---------------------------------------------------------------------------------------------
# Orchestrator (routing + latency budget)
# ---------------------------------------------------------------------------------------------
LATENCY_BUDGET_S = float(os.environ.get("HALLU_BUDGET_S", 90))   # total per query on CPU (generation dominates)
RESERVE_PER_CLAIM_S = 0.6       # minimum time that must remain to start an NLI check on one more claim
WEAK_RETRIEVAL_T = 0.80         # retrieval top cosine below this raises the risk score (e5 scores sit ~0.75-0.9)
EARLY_EXIT_T = float(os.environ.get("HALLU_EARLY_EXIT", 0.0))   # top cosine below this -> skip the LLM, answer NOT_FOUND.
                                # 0 = off. Calibrate with `python evaluate_e2e.py` (prints a suggested value), then set it.
LOW_RISK_T = 0.25               # pre-check risk below this -> fast path (tier-0 + tier-1 on numeric claims only)
POLICIES = ["adaptive", "always_full", "heuristic_only", "no_verify"]   # adaptive = proposed; others = baselines
DEFAULT_POLICY = "adaptive"
