# Setup Guide: Resource-Constrained Heterogeneous Multi-Agent Framework (Oct + Nov milestones)

Offline, multilingual (Marathi + English) policy question answering with a retriever and a locally hosted
quantised Llama generator. Hallucination verification (Verifier / Critic / Orchestrator) comes in later milestones.

## 1. What this project does
    Question (Marathi/English)
        -> Retriever Agent : multilingual-e5-small embeddings + FAISS index over 6,438 GR chunks (300 Maharashtra GRs)
        -> Generator Agent : quantised Llama (GGUF, llama.cpp, CPU only) answers ONLY from the retrieved evidence
        -> Output          : answer, sources (chunk id, page, score), NOT_FOUND abstention, per-stage latency, run log

| Milestone | Deliverable | Files |
|---|---|---|
| October | Corpus, cleaning, chunking, local FAISS vector index, Retriever Agent | data/chunks.jsonl, common.py, build_index.py, retriever_agent.py |
| November | Locally hosted quantised Llama as Generator Agent, integrated with Retriever for evidence-grounded answers, evaluation | nov_config.py, download_model.py, generator_agent.py, pipeline.py, evaluate_nov.py, eval_queries.json |

## 2. Requirements
- Windows 10/11 64-bit, Python 3.10, 3.11 or 3.12 (64-bit, "Add to PATH" ticked), VS Code
- ~15 GB free disk (libraries ~4 GB, models 2-5 GB, index ~10 MB)
- Internet for the FIRST setup only. After that everything runs offline.
- RAM guide: **8 GB -> llama-3.2-3b** (default).  **16 GB -> llama-3.1-8b** (the Llama-3 8B class model of the synopsis).

## 3. Fast path (recommended): one script
Open the project folder in VS Code, open a PowerShell terminal (Ctrl + `), then:

    Set-ExecutionPolicy -Scope Process Bypass
    .\setup.ps1                        # 8 GB laptop
    .\setup.ps1 -Model llama-3.1-8b    # 16 GB laptop

It creates the virtual environment, installs libraries and the llama-cpp wheel, builds the FAISS index,
downloads the model and runs two smoke tests. Expect 30-60 minutes (mostly downloads + embedding).

### Shortcut: copy instead of rebuilding
If you have a working laptop, copy these folders to the new one (pen drive) before running setup.ps1:
- `index\`  (skips the 5-20 min embedding step)
- `models\` (skips the model download; setup.ps1 still checks it)
- For a fully offline new laptop also copy the embedding-model cache:
  `C:\Users\<you>\.cache\huggingface\hub\models--intfloat--multilingual-e5-small`  to the same path on the new laptop.

## 4. Manual path (what setup.ps1 does, step by step)
    python -m venv .venv
    .\.venv\Scripts\Activate.ps1
    pip install -r requirements.txt
    pip install llama-cpp-python --only-binary=:all: --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
    python build_index.py
    python download_model.py                  # or: python download_model.py llama-3.1-8b
    python pipeline.py "अंगणवाडी सेविकांना मानधन किती मिळते?"

## 5. Everyday use
    .\.venv\Scripts\Activate.ps1
    python retriever_agent.py "Birsa Munda Krishi Kranti Yojana subsidy" -k 3 --lang mr
    python pipeline.py "बिरसा मुंडा कृषि क्रांती योजनेत नवीन सिंचन विहिरीसाठी अनुदान किती आहे?"
    python pipeline.py --chat
    python pipeline.py "..." --model llama-3.1-8b      # choose model (16 GB laptop)
    python evaluate_nov.py --no-gen                    # retrieval metrics only (fast)
    python evaluate_nov.py                             # full evaluation (3B: ~12-15 min; 8B slower)
Set a default model for the session:  `$env:HALLU_MODEL = "llama-3.1-8b"`

Outputs for the thesis: `logs/runs.jsonl` (every query), `logs/eval_results.jsonl`, `logs/eval_summary.json`.

## 6. Prove it is offline (for the demo)
    $env:HF_HUB_OFFLINE = 1
    python pipeline.py "अंगणवाडी सेविकांना मानधन किती मिळते?"
    Remove-Item Env:HF_HUB_OFFLINE          # needed again before any download
Or simply turn Wi-Fi off. Retrieval and generation must still work.

## 7. Suggested presentation flow (5 minutes)
1. `python retriever_agent.py "<Marathi query>" -k 3`  -> show chunks, pages, scores, latency (milliseconds).
2. `python pipeline.py "<same query>"`  -> show grounded answer + sources + timing split (retrieval vs generation).
3. `python pipeline.py "What is the capital of France?"` -> shows abstention (NOT_FOUND) on out-of-scope input.
4. Show `logs/eval_summary.json` (Hit@1, Hit@3, MRR, abstention rate, latency).
5. State the findings honestly: retrieval is strong; the small quantised model can still pick a wrong number from a
   correct document (e.g. a table cell) -> this is exactly what the Verifier Agent (NLI) will detect next.

## 8. Troubleshooting
| Symptom | Fix |
|---|---|
| `running scripts is disabled` | `Set-ExecutionPolicy -Scope Process Bypass` then re-run |
| `OfflineModeIsEnabled` while downloading | `Remove-Item Env:HF_HUB_OFFLINE` |
| Download seems stuck at "reconstructing file" | wait a few minutes; if size does not grow: Ctrl+C, run again (it resumes); or `$env:HF_HUB_DISABLE_XET=1` |
| `pip install torch --index-url ...` fails | not needed; `pip install -r requirements.txt` installs a working torch |
| llama-cpp-python tries to compile / error | always use `--only-binary=:all:` with the extra-index-url given above |
| `logprobs is not supported ...` | already fixed in this version (confidence is `None` for now) |
| Very slow generation (~50 s on 3B) | normal on CPU: Marathi needs many tokens. Close other apps; keep `-k 3` |
| System freezes with 8B on 8 GB RAM | use `llama-3.2-3b`; 8B needs ~6 GB free RAM |
| `git push` fails with big file | make sure `.gitignore` contains `models/` (included) |

## 9. Known limitations (state these in the report)
- Llama 3.x officially supports Hindi but not Marathi; Marathi answers can be wrong or malformed.
- English chunks are machine translations of Marathi GRs and are noisy; evidence is therefore taken from the Marathi chunks.
- Tables extracted from PDFs are flattened, so numbers can be attributed to the wrong column.
- Generator confidence score is not implemented yet (needs a memory-light method).
- Evaluation set is small (14 queries): report results as preliminary.
