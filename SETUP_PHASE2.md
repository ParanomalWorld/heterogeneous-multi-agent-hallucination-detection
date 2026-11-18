# Phase 2 — Verifier, Critic, Orchestrator, Web App, Evaluation (Dec 2026 – Mar 2027 milestones)

This continues the same `oct_task` project. October (FAISS retriever) and November (Llama generator) stay as they are.
Phase 2 adds the remaining three agents, the latency budget, a local web application and the evaluation that
produces the tables/figures for the research paper.

```
Question (मराठी / English)
  -> Orchestrator  (LangGraph)  ...................... routes on retrieval score, LLM confidence, claim risk, time left
  -> Retriever     e5-small + FAISS ................... (October)
  -> Generator     Llama GGUF, streamed, confidence ... (November + token confidence)
  -> Verifier      Tier-0 heuristic -> Tier-1 mDeBERTa NLI (int8) -> Tier-2 deep NLI (only if uncertain + time left)
  -> Critic        error type per claim + verdict (Verified / Partially / Hallucinated / Unverified / Abstained)
  -> Web UI        streamed answer, coloured claims, evidence, decision trace, audit JSON
```

## New files
| File | Agent / purpose |
|---|---|
| `hallu_config.py` | every threshold of phase 2 (reported in the paper's setup table) |
| `text_utils.py` | claim splitting, Devanagari digits, Indian number format, premise selection (Tier-0) |
| `verifier_agent.py` | Verifier Agent: heterogeneous NLI cascade |
| `critic_agent.py` | Critic Agent: hallucination taxonomy, risk, verdict |
| `orchestrator.py` | Orchestrator Agent + Latency Budget Monitor + full graph (CLI) |
| `app.py`, `web/index.html` | local web application (FastAPI, no internet, no CDN) |
| `hallu_bench.py` | MH-PolicyHalluBench + policy ablation + budget sweep -> paper tables & figures |
| `evaluate_e2e.py` | end-to-end evaluation with the real LLM + human annotation sheet |
| `download_nli.py` | one-time download of the NLI models |
| `tests/smoke_test.py` | 10-second self test with mock models |
| `run_app.ps1` | start the web app |
| `generator_agent.py` (updated) | streaming tokens + token-probability confidence (no extra RAM) |

## Step-by-step in VS Code (Windows)

Open the folder in VS Code → Terminal → New Terminal (PowerShell). Do the steps in order; each has a check.

### Step 1 — Activate the existing environment
```powershell
cd C:\Users\jayas\Downloads\oct_task_full\oct_task
.\.venv\Scripts\Activate.ps1
```
No `.venv` yet on this laptop? Run `Set-ExecutionPolicy -Scope Process Bypass ; .\setup.ps1` first (October/November setup).

### Step 2 — Install the phase-2 libraries (internet, once, ~5 min)
```powershell
Remove-Item Env:HF_HUB_OFFLINE -ErrorAction SilentlyContinue
pip install -r requirements.txt
```
Check: `python -c "import transformers, langgraph, fastapi; print('ok')"` prints `ok`.

### Step 3 — Self-test with mock models (no downloads needed)
```powershell
python tests/smoke_test.py
```
Check: `ALL PASS`. (Needs `index\` from October. If missing: `python build_index.py`.)

### Step 4 — Download the Verifier model (internet, once)
```powershell
python download_nli.py              # 8 GB laptop: Tier-1 only (~1.1 GB)
python download_nli.py --tier2      # 16 GB laptop: also the large DeBERTa (~1.7 GB)
python verifier_agent.py
```
Check: the first demo claim (15,000 – wrong number) is `CONTRADICTED` / `UNCERTAIN`, the second (5,500) is `SUPPORTED`.

### Step 5 — Run the full 5-agent pipeline from the terminal
```powershell
python orchestrator.py "अंगणवाडी सेविकांना मानधन किती मिळते?" --no-tier2
python orchestrator.py "What subsidy is given for a new irrigation well under Birsa Munda Krishi Kranti Yojana?" --no-tier2
python orchestrator.py "What is the capital of France?" --no-tier2           # should abstain
python orchestrator.py "अंगणवाडी सेविकांना मानधन किती मिळते?" --budget 5    # tiny budget -> "UNVERIFIED"
```
Check: a VERDICT line, coloured claims list and a decision trace are printed. Every run is appended to
`logs\orchestrator_runs.jsonl`.

### Step 6 — Start the web application
```powershell
.\run_app.ps1                       # 8 GB:  3B model, Tier-1 NLI
.\run_app.ps1 -Model llama-3.1-8b -Tier2    # 16 GB
.\run_app.ps1 -Mock                 # UI only, any laptop, instant
```
The browser opens http://127.0.0.1:8000. The status pill turns green when models are loaded (~1 min).
`run_app.ps1` sets `HF_HUB_OFFLINE=1`, so you can switch Wi-Fi off during the demo.

### Step 7 — Main research results: benchmark + ablation (no LLM needed, ~20-40 min on CPU)
```powershell
python hallu_bench.py --n 150 --no-tier2      # 8 GB
python hallu_bench.py --n 150                 # 16 GB (with Tier-2)
```
Outputs in `results\`: `bench_tables.md` (Tables A-C, paste into paper), `fig_policy_tradeoff.png`,
`fig_budget_sweep.png`, `bench_summary.json`. The web app's **Evaluation** tab shows them too.
Quick plumbing test: `python hallu_bench.py --n 20 --mock-nli` (numbers are meaningless in mock mode).

### Step 8 — End-to-end evaluation with the real LLM (~15-30 min)
```powershell
python evaluate_e2e.py --no-tier2                     # 16 GB: python evaluate_e2e.py --model llama-3.1-8b
```
Then open `results\e2e_annotation.csv` in Excel, fill `human_label` for every claim
(SUPPORTED / NUMERIC_FABRICATION / CONTRADICTION / UNSUPPORTED), save, and run:
```powershell
python evaluate_e2e.py --score results\e2e_annotation.csv
```
→ detector precision/recall against human judgement on REAL LLM answers (the strongest table for reviewers).
It also prints `suggested_EARLY_EXIT_T`; put that value in `hallu_config.py` to skip the LLM for out-of-scope questions.

### Step 9 — Git (keep the milestone branches)
```powershell
git checkout -b dec-2026
git add .
git commit -m "Phase 2: verifier cascade, critic, orchestrator, web app, benchmark"
git push -u origin dec-2026
```

## RAM plan
| Laptop | LLM | Tier-1 NLI | Tier-2 NLI | Peak RAM (approx.) |
|---|---|---|---|---|
| 8 GB | llama-3.2-3b Q4 (~2.5 GB) | mDeBERTa-base int8 (~0.4 GB) | off (`--no-tier2`) | ~4.5 GB |
| 16 GB | llama-3.1-8b Q4 (~5.5 GB) | mDeBERTa-base int8 | DeBERTa-v3-large int8 (~0.6 GB) | ~8 GB |

## Troubleshooting
| Symptom | Fix |
|---|---|
| Status pill shows `error: ... not found` | run `python download_model.py` / `python download_nli.py` |
| `OfflineModeIsEnabled` while downloading | `Remove-Item Env:HF_HUB_OFFLINE` |
| Web page says "Models not ready" | wait for the green pill (first load ~1 min) |
| Everything is slow | close Chrome tabs; use `--no-tier2`; lower `MAX_NEW_TOKENS` in `nov_config.py` |
| `langgraph` import error | the built-in runner is used automatically (same graph); `pip install langgraph` to use LangGraph |
| Port 8000 busy | `.\run_app.ps1 -Port 8010` |

## Honest notes for the viva / paper
- `DeBERTa-v3-large-MNLI` (synopsis) is trained on English. The GR evidence is Marathi, so Tier-1 is the
  multilingual mDeBERTa-v3 XNLI model and the large English model is used only for English–English pairs.
  Present this as a finding of the work, not a hidden change.
- The 8 GB laptop uses Llama-3.2-3B; the 16 GB laptop runs the Llama-3.1-8B (Llama-3 8B class) of the synopsis.
- The "inject hallucination" option in the web app is a labelled demo/test mode, used to show detection live.
