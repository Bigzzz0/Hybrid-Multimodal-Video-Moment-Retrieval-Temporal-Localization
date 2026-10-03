# ==============================================================================
#   Pure-Visual Video Moment Retrieval and Temporal Localization System
#   PowerShell Launcher (Backend + AI Preload + Worker + Frontend)
# ==============================================================================

$RootDir = $PSScriptRoot
$BackendDir = Join-Path $RootDir "backend"
$FrontendDir = Join-Path $RootDir "frontend"
$PythonExe = Join-Path $BackendDir "venv\Scripts\python.exe"

Write-Host "==============================================================================" -ForegroundColor Cyan
Write-Host "  Pure-Visual Video Moment Retrieval & Temporal Localization System" -ForegroundColor Cyan
Write-Host "  One-Click Unified Launcher (AI Preload + Backend + Worker + Frontend)" -ForegroundColor Cyan
Write-Host "==============================================================================" -ForegroundColor Cyan
Write-Host ""

# 1. Verify Virtual Environment
if (-not (Test-Path $PythonExe)) {
    Write-Host "[ERROR] Virtual environment not found at: $PythonExe" -ForegroundColor Red
    Write-Host "Please create the virtual environment first:"
    Write-Host "  cd backend"
    Write-Host "  python -m venv venv"
    Write-Host "  .\venv\Scripts\activate"
    Write-Host "  pip install -r requirements.txt"
    exit 1
}

# 2. Verify Frontend Dependencies
$NodeModules = Join-Path $FrontendDir "node_modules"
if (-not (Test-Path $NodeModules)) {
    Write-Host "[INFO] Installing frontend dependencies (npm install)..." -ForegroundColor Yellow
    Push-Location $FrontendDir
    npm install
    Pop-Location
}

# 3. Check for .env file
$EnvFile = Join-Path $BackendDir ".env"
$EnvExample = Join-Path $BackendDir ".env.example"
if (-not (Test-Path $EnvFile) -and (Test-Path $EnvExample)) {
    Write-Host "[INFO] Creating backend/.env from .env.example..." -ForegroundColor Yellow
    Copy-Item $EnvExample $EnvFile
}

# 4. Automated AI Model Preloading (Zero Cold-Start)
Write-Host ""
Write-Host "[1/4] Preloading AI Models (SigLIP 2 Vision & Captioner)..." -ForegroundColor Cyan
Push-Location $BackendDir
& $PythonExe preload_models.py
Pop-Location
Write-Host "[OK] AI Models preloaded and ready in memory." -ForegroundColor Green
Write-Host ""

# 5. Launch Backend API Server (FastAPI on Port 8000)
Write-Host "[2/4] Starting Backend API Server (http://localhost:8000)..." -ForegroundColor Green
Start-Process cmd.exe -ArgumentList "/k cd /d `"$BackendDir`" && .\venv\Scripts\python.exe main.py" -WindowStyle Normal

# 6. Launch Inference Worker (if enabled in .env)
$WorkerEnabled = $false
if (Test-Path $EnvFile) {
    $EnvContent = Get-Content $EnvFile
    if ($EnvContent -match "INFERENCE_WORKER_ENABLED=true") {
        $WorkerEnabled = $true
    }
}

if ($WorkerEnabled) {
    Write-Host "[3/4] Starting Inference Worker (http://localhost:8011)..." -ForegroundColor Green
    Start-Process cmd.exe -ArgumentList "/k cd /d `"$BackendDir`" && .\venv\Scripts\python.exe -m inference_worker.main" -WindowStyle Normal
} else {
    Write-Host "[3/4] Inference worker disabled (INFERENCE_WORKER_ENABLED=false). Running Fast Retrieval." -ForegroundColor Gray
}

# 7. Launch Frontend Server (Next.js on Port 3000)
Write-Host "[4/4] Starting Frontend Web Server (http://localhost:3000)..." -ForegroundColor Green
Start-Process cmd.exe -ArgumentList "/k cd /d `"$FrontendDir`" && npm run dev" -WindowStyle Normal

Write-Host ""
Write-Host "==============================================================================" -ForegroundColor Cyan
Write-Host "  All services have been launched successfully!" -ForegroundColor Green
Write-Host ""
Write-Host "  Web Application UI : http://localhost:3000" -ForegroundColor Yellow
Write-Host "  Backend API Docs   : http://localhost:8000/docs" -ForegroundColor Yellow
Write-Host ""
Write-Host "  To stop all services:" -ForegroundColor White
Write-Host "    - Close the individual command windows, OR"
Write-Host "    - Run './stop_all.ps1' or 'stop_all.bat'"
Write-Host "==============================================================================" -ForegroundColor Cyan
Write-Host ""

# 8. Wait 5 seconds and open browser automatically
Write-Host "Opening browser in 5 seconds..." -ForegroundColor Gray
Start-Sleep -Seconds 5
Start-Process "http://localhost:3000"
