$ErrorActionPreference = 'Stop'
$experimentRoot = Split-Path -Parent $PSScriptRoot
$experimentPython = Join-Path $experimentRoot '.venv-embeddinggemma2/Scripts/python.exe'
$experimentFrontend = Join-Path $experimentRoot 'frontend'
$experimentNext = Join-Path $experimentFrontend 'node_modules/next/dist/bin/next'
$experimentLogs = Join-Path $experimentRoot 'scratch/embeddinggemma2/ui/logs'
if (!(Test-Path -LiteralPath $experimentPython) -or !(Test-Path -LiteralPath $experimentNext)) {
    throw 'Run the experiment setup and frontend npm install first.'
}
New-Item -ItemType Directory -Path $experimentLogs -Force | Out-Null
if (!(Get-NetTCPConnection -LocalPort 8012 -State Listen -ErrorAction SilentlyContinue)) {
    Start-Process -FilePath $experimentPython -ArgumentList @('-m', 'uvicorn', 'evaluation.embeddinggemma2.service:app', '--host', '127.0.0.1', '--port', '8012') -WorkingDirectory $experimentRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $experimentLogs 'api.log') -RedirectStandardError (Join-Path $experimentLogs 'api-error.log')
}
if (!(Get-NetTCPConnection -LocalPort 3000 -State Listen -ErrorAction SilentlyContinue)) {
    $experimentNode = (Get-Command node.exe).Source
    Start-Process -FilePath $experimentNode -ArgumentList @(('"' + $experimentNext + '"'), 'dev', '--hostname', '127.0.0.1', '--port', '3000') -WorkingDirectory $experimentFrontend -WindowStyle Hidden -RedirectStandardOutput (Join-Path $experimentLogs 'frontend.log') -RedirectStandardError (Join-Path $experimentLogs 'frontend-error.log')
}
Write-Host 'UI: http://127.0.0.1:3000/experiments/embeddinggemma2'
Write-Host 'Services load model weights only after you press Search. Logs:' $experimentLogs
