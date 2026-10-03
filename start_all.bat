@echo off
setlocal EnableDelayedExpansion

title Pure-Visual Video Moment Retrieval Launcher

echo ==============================================================================
echo   Pure-Visual Video Moment Retrieval and Temporal Localization System
echo   One-Click Unified Launcher (AI Preload + Backend + Worker + Frontend)
echo ==============================================================================
echo.

set "ROOT_DIR=%~dp0"
set "BACKEND_DIR=%ROOT_DIR%backend"
set "FRONTEND_DIR=%ROOT_DIR%frontend"
set "PYTHON_EXE=%BACKEND_DIR%\venv\Scripts\python.exe"

REM 1. Verify Virtual Environment
if not exist "%PYTHON_EXE%" (
    echo [ERROR] Virtual environment not found at: %PYTHON_EXE%
    echo Please create the virtual environment first:
    echo   cd backend
    echo   python -m venv venv
    echo   .\venv\Scripts\activate
    echo   pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

REM 2. Verify Frontend Dependencies
if not exist "%FRONTEND_DIR%\node_modules" (
    echo [INFO] Installing frontend dependencies (npm install)...
    cd /d "%FRONTEND_DIR%"
    call npm.cmd install
    if errorlevel 1 (
        echo [ERROR] Failed to install frontend dependencies.
        pause
        exit /b 1
    )
    cd /d "%ROOT_DIR%"
)

REM 3. Check for .env file
if not exist "%BACKEND_DIR%\.env" (
    if exist "%BACKEND_DIR%\.env.example" (
        echo [INFO] Creating backend\.env from .env.example...
        copy "%BACKEND_DIR%\.env.example" "%BACKEND_DIR%\.env" >nul
    )
)

REM 4. Automated AI Model Preloading (Zero Cold-Start)
echo.
echo [1/4] Preloading AI Models (SigLIP 2 Vision & Captioner)...
cd /d "%BACKEND_DIR%"
"%PYTHON_EXE%" preload_models.py
if errorlevel 1 (
    echo [WARN] Model preloading encountered an issue, proceeding to server startup...
) else (
    echo [OK] AI Models preloaded and ready in memory.
)
cd /d "%ROOT_DIR%"
echo.

REM 5. Launch Backend API Server (FastAPI on Port 8000)
echo [2/4] Starting Backend API Server (http://localhost:8000)...
start "Moment Retrieval - Backend API (:8000)" cmd /k "cd /d "%BACKEND_DIR%" && .\venv\Scripts\python.exe main.py"

REM 6. Launch Inference Worker (if enabled in .env)
findstr /i "INFERENCE_WORKER_ENABLED=true" "%BACKEND_DIR%\.env" >nul 2>&1
if not errorlevel 1 (
    echo [3/4] Starting Inference Worker (http://localhost:8011)...
    start "Moment Retrieval - AI Worker (:8011)" cmd /k "cd /d "%BACKEND_DIR%" && .\venv\Scripts\python.exe -m inference_worker.main"
) else (
    echo [3/4] Inference worker disabled (INFERENCE_WORKER_ENABLED=false). Running Fast Retrieval.
)

REM 7. Launch Frontend Server (Next.js on Port 3000)
echo [4/4] Starting Frontend Web Server (http://localhost:3000)...
start "Moment Retrieval - Frontend Web (:3000)" cmd /k "cd /d "%FRONTEND_DIR%" && npm.cmd run dev"

echo.
echo ==============================================================================
echo   All services have been launched successfully!
echo.
echo   Web Application UI : http://localhost:3000
echo   Backend API Docs   : http://localhost:8000/docs
echo.
echo   To stop all services:
echo     - Close the individual command windows, OR
echo     - Double-click 'stop_all.bat' from this directory
echo ==============================================================================
echo.

REM 8. Wait 5 seconds and open browser automatically
echo Opening web browser at http://localhost:3000 in 5 seconds...
timeout /t 5 /nobreak >nul
start http://localhost:3000
