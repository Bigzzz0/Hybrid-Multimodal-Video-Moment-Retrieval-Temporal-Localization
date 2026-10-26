@echo off
setlocal EnableDelayedExpansion
title Stop Moment Retrieval Services

echo ==============================================================================
echo   Stopping Pure-Visual Video Moment Retrieval Services...
echo ==============================================================================
echo.

for %%P in (8000 8011 3000 18100 18101) do (
    for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":%%P" ^| findstr "LISTENING"') do (
        echo [STOP] Terminating process on port %%P - PID: %%a
        taskkill /F /PID %%a >nul 2>&1
    )
)
taskkill /F /IM llama-server.exe >nul 2>&1

echo.
echo ==============================================================================
echo   [OK] All services on ports 8000, 8011, 3000, 18100, 18101 stopped.
echo ==============================================================================
echo.
timeout /t 3 >nul 2>&1
