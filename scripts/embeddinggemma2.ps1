param(
    [switch]$Setup,
    [switch]$Run
)
$ErrorActionPreference = 'Stop'
if ($Setup -and $Run) { throw 'Use -Setup or -Run separately.' }
$taskRoot = Split-Path -Parent $PSScriptRoot
$taskPython = Join-Path $taskRoot '.venv-embeddinggemma2/Scripts/python.exe'
Push-Location $taskRoot
try {
    $env:PYTHONIOENCODING = 'utf-8'
    if ($Setup) {
        $taskBasePython = Join-Path $taskRoot '.venv-inference/Scripts/python.exe'
        if (-not (Test-Path -LiteralPath $taskBasePython)) { throw 'Prepare the existing CUDA inference environment first.' }
        if (-not (Test-Path -LiteralPath $taskPython)) {
            & $taskBasePython -m venv '.venv-embeddinggemma2'
            if ($LASTEXITCODE -ne 0) { throw 'Could not create experiment environment.' }
        }
        $taskBaseSite = Join-Path $taskRoot '.venv-inference/Lib/site-packages'
        $taskPth = Join-Path $taskRoot '.venv-embeddinggemma2/Lib/site-packages/_experiment_base.pth'
        [System.IO.File]::WriteAllText($taskPth, $taskBaseSite + [Environment]::NewLine, (New-Object System.Text.UTF8Encoding($false)))
        & $taskPython -m pip install --no-deps -r 'evaluation/embeddinggemma2/requirements.txt'
        if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
        & $taskPython -u -m evaluation.embeddinggemma2.prepare --download
    } elseif ($Run) {
        if (-not (Test-Path -LiteralPath $taskPython)) { throw 'Run setup first.' }
        & $taskPython -u -m evaluation.embeddinggemma2.benchmark --run
    } else {
        if (-not (Test-Path -LiteralPath $taskPython)) { throw 'Run setup first.' }
        & $taskPython -u -m evaluation.embeddinggemma2.prepare
    }
    if ($LASTEXITCODE -ne 0) { throw "Experiment command failed with exit code $LASTEXITCODE" }
} finally {
    Pop-Location
}
