# NitiSatya - Resource-Constrained Heterogeneous Multi-Agent Hallucination Detection (Maharashtra policy QA)
M.Tech project, GHRCE Nagpur (Centre of Excellence - Digital Technology). Offline, CPU-only, Marathi + English.

- October / November (retriever + generator): see SETUP.md, PROMPTS.md
- Phase 2 (verifier, critic, orchestrator, web app, evaluation): see **SETUP_PHASE2.md**

Quick start (Windows PowerShell, after setup):
    .\.venv\Scripts\Activate.ps1
    python tests/smoke_test.py          # self test (mock models)
    .\run_app.ps1                       # web app on http://127.0.0.1:8000
    python hallu_bench.py --n 150 --no-tier2   # paper tables + figures in results/
