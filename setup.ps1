<# One-shot setup for a fresh Windows laptop (all phases: retriever, generator, verifier, web app).
   Run in PowerShell from the project folder:
       Set-ExecutionPolicy -Scope Process Bypass
       .\setup.ps1                                # 8 GB RAM  -> Llama-3.2-3B, Tier-1 NLI
       .\setup.ps1 -Model llama-3.1-8b -Tier2     # 16 GB RAM -> Llama-3.1-8B, Tier-1 + Tier-2 NLI
   Needs internet once (~4-8 GB downloads). Afterwards everything runs offline. Safe to re-run. #>
param([string]$Model = "llama-3.2-3b", [switch]$Tier2)
$ErrorActionPreference = "Stop"
Remove-Item Env:HF_HUB_OFFLINE -ErrorAction SilentlyContinue

function Step($t) { Write-Host "`n=== $t ===" -ForegroundColor Cyan }
function Check($what) { if ($LASTEXITCODE -ne 0) { throw "FAILED: $what (see message above)" } }

Step "1/8 Python check (needs 3.10 - 3.12, 64-bit)"
python --version; Check "python not found - install Python 3.10-3.12 (64-bit) and tick 'Add to PATH'"

Step "2/8 Virtual environment (.venv)"
if (-not (Test-Path ".venv")) { python -m venv .venv; Check "venv" }
. .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip -q

Step "3/8 Python libraries (torch, faiss, transformers, langgraph, fastapi ...)"
pip install -r requirements.txt; Check "pip install -r requirements.txt"

Step "4/8 llama-cpp-python (prebuilt CPU wheel, no compiler needed)"
pip install llama-cpp-python --only-binary=:all: --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
Check "llama-cpp-python wheel"
python -c "from llama_cpp import Llama; print('llama-cpp OK')"; Check "llama-cpp import"

Step "5/8 FAISS vector index (skipped if index\ already exists; else 5-20 min)"
if (-not (Test-Path "index\build_info.json")) { python build_index.py; Check "build_index.py" }
else { Write-Host "index\ found - skipping" }

Step "6/8 Language model ($Model)"
python download_model.py $Model; Check "download_model.py"

Step "7/8 Verifier NLI model(s)"
if ($Tier2) { python download_nli.py --tier2 } else { python download_nli.py }
Check "download_nli.py"

Step "8/8 Self tests"
python tests/smoke_test.py; Check "smoke test"
python nli_selftest.py; Check "NLI self test"

Write-Host "`nSETUP COMPLETE." -ForegroundColor Green
Write-Host "Start the web app:   .\run_app.ps1 -Model $Model $(if ($Tier2) {'-Tier2'})"
Write-Host "Terminal demo:       python orchestrator.py `"What is the new honorarium of Anganwadi workers?`" --model $Model $(if (-not $Tier2) {'--no-tier2'})"
