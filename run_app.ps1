<# Start the local web app.   .\run_app.ps1            (8 GB laptop, 3B model, no Tier-2)
                              .\run_app.ps1 -Model llama-3.1-8b -Tier2     (16 GB laptop)
                              .\run_app.ps1 -Mock      (UI demo without LLM / NLI; still needs index\) #>
param([string]$Model = "llama-3.2-3b", [switch]$Tier2, [switch]$Mock, [int]$Port = 8000)
# Use .venv only if the project's packages are installed there; otherwise the global Python is used.
if (Test-Path ".\.venv\Scripts\python.exe") {
    & .\.venv\Scripts\python.exe -c "import uvicorn, faiss, transformers, llama_cpp" 2>$null
    if ($LASTEXITCODE -eq 0) { . .\.venv\Scripts\Activate.ps1 }
    else { if (Get-Command deactivate -ErrorAction SilentlyContinue) { deactivate }
           Write-Host "(.venv is empty - using global Python)" -ForegroundColor Yellow }
}
if (-not (Test-Path "index\build_info.json")) {
    Write-Host "index\ not found. Run:  python build_index.py   (or copy the index folder from the October laptop)" -ForegroundColor Red
    exit 1
}
$oldOffline = $env:HF_HUB_OFFLINE
$env:HF_HUB_OFFLINE = 1          # prove it: no internet used at runtime (only while the app runs)
$a = @("app.py", "--model", $Model, "--port", $Port)
if (-not $Tier2) { $a += "--no-tier2" }
if ($Mock) { $a += "--mock" }
Start-Process "http://127.0.0.1:$Port"
try { python @a } finally { if ($null -eq $oldOffline) { Remove-Item Env:HF_HUB_OFFLINE -ErrorAction SilentlyContinue } else { $env:HF_HUB_OFFLINE = $oldOffline } }
