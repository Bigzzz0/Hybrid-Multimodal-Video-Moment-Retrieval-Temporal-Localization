param(
    [string]$SourceCheckout = 'C:\Users\User\Downloads\Video Event Retrieval',
    [switch]$NoFrontend
)
$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$sourceRepo = (Resolve-Path $SourceCheckout).Path
$gemmaPython = Join-Path $sourceRepo '.venv-embeddinggemma2\Scripts\python.exe'
$qwenPython = Join-Path $sourceRepo '.venv-inference\Scripts\python.exe'
foreach ($pythonPath in @($gemmaPython, $qwenPython)) {
    if (!(Test-Path -LiteralPath $pythonPath)) { throw "Missing isolated runtime: $pythonPath" }
}
$env:PYTHONPATH = "$repo\backend;$repo"
& $gemmaPython -m evaluation.embeddinggemma2.check_runtime
if ($LASTEXITCODE -ne 0) { throw 'Gemma dependency pins do not match; production environment was not changed.' }
$env:GEMMA_MODEL_ROOT = Join-Path $sourceRepo 'scratch\embeddinggemma2\models'
$env:GEMMA_DATA_DIR = Join-Path $repo 'backend\data\gemma2'
$env:GEMMA_WORKER_URL = 'http://127.0.0.1:8012'
$env:GEMMA_QWEN_URL = 'http://127.0.0.1:8013'
$env:INFERENCE_WORKER_URL = 'http://127.0.0.1:8013'
$env:RETRIEVAL_BACKEND = 'gemma2'
$env:VLM_GGUF_DIR = Join-Path $sourceRepo 'models\caprl-qwen3vl-4b-gguf'
$env:LLAMA_CPP_PATH = Join-Path $sourceRepo 'models\llama-b10964\llama.exe'
$env:INFERENCE_REQUEST_TIMEOUT_SEC = '240'
$env:INFERENCE_WARMUP_QWEN = 'false'
$env:NEXT_PUBLIC_API_URL = 'http://127.0.0.1:8002'
$logDir = Join-Path $repo 'scratch\event-retrieval\logs'
New-Item -ItemType Directory -Path $logDir -Force | Out-Null
$launched = @()
function Start-LocalService($port, $pythonPath, $module, $name) {
    if (Get-NetTCPConnection -State Listen -LocalPort $port -ErrorAction SilentlyContinue) {
        Write-Host "$name already listening on $port; verify /health before using it."
        return
    }
    $process = Start-Process -FilePath $pythonPath -ArgumentList @('-m', 'uvicorn', $module, '--host', '127.0.0.1', '--port', "$port") -WorkingDirectory $repo -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $logDir "$name.stdout.log") -RedirectStandardError (Join-Path $logDir "$name.stderr.log")
    $script:launched += @{ name=$name; pid=$process.Id; port=$port }
}
Start-LocalService 8013 $qwenPython 'inference_worker.main:app' 'qwen'
Start-LocalService 8012 $gemmaPython 'evaluation.embeddinggemma2.worker:app' 'gemma'
Start-LocalService 8002 $qwenPython 'main:app' 'api'
if (!$NoFrontend) {
    $modules = Join-Path $repo 'frontend\node_modules'
    if (!(Test-Path -LiteralPath $modules)) {
        $sharedModules = Join-Path $sourceRepo 'frontend\node_modules'
        if (!(Test-Path -LiteralPath $sharedModules)) { throw 'Run npm ci in the branch frontend first.' }
        New-Item -ItemType Junction -Path $modules -Target $sharedModules | Out-Null
    }
    if (!(Get-NetTCPConnection -State Listen -LocalPort 3002 -ErrorAction SilentlyContinue)) {
        $next = Join-Path $modules 'next\dist\bin\next'
        $process = Start-Process -FilePath 'node.exe' -ArgumentList @("`"$next`"", 'dev', '-H', '127.0.0.1', '-p', '3002') -WorkingDirectory (Join-Path $repo 'frontend') -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $logDir 'frontend.stdout.log') -RedirectStandardError (Join-Path $logDir 'frontend.stderr.log')
        $launched += @{ name='frontend'; pid=$process.Id; port=3002 }
    }
}
$launched | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $logDir 'started-processes.json')
function Wait-LocalHealth($port, $path, $kind) {
    $watch = [System.Diagnostics.Stopwatch]::StartNew()
    while ($watch.Elapsed.TotalSeconds -lt 30) {
        try {
            $health = Invoke-RestMethod -Uri "http://127.0.0.1:$port$path" -TimeoutSec 3
            $matches = switch ($kind) {
                'qwen' { $health.worker -eq 'local' }
                'gemma' { $health.model_id -eq 'google/embeddinggemma-2' -and $health.revision -eq '914f7f89142e33e77833254d9c9b90c3cef7303b' }
                'api' { $health.backend -eq 'gemma2' }
            }
            if ($matches) { return }
        } catch { }
        Start-Sleep -Milliseconds 200
    }
    throw "$kind on $port is unavailable or belongs to a different service. Check $logDir"
}
Wait-LocalHealth 8013 '/health' 'qwen'
Wait-LocalHealth 8012 '/health' 'gemma'
Wait-LocalHealth 8002 '/api/v1/events/health' 'api'
Write-Host 'Branch demo ready: http://127.0.0.1:3002/events. No indexing starts automatically.'
