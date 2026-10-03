# ==============================================================================
#   Stop Pure-Visual Video Moment Retrieval Services
# ==============================================================================

Write-Host "==============================================================================" -ForegroundColor Yellow
Write-Host "  Stopping Pure-Visual Video Moment Retrieval Services..." -ForegroundColor Yellow
Write-Host "==============================================================================" -ForegroundColor Yellow
Write-Host ""

$Ports = @(8000, 8011, 3000)

foreach ($Port in $Ports) {
    $Lines = netstat -aon | Select-String ":$Port\s+.*LISTENING"
    foreach ($Line in $Lines) {
        $Parts = $Line.ToString().Trim() -split "\s+"
        $PidVal = $Parts[-1]
        if ($PidVal -match "^\d+$" -and $PidVal -ne "0") {
            Write-Host "[STOP] Terminating process on port $Port (PID: $PidVal)..." -ForegroundColor Red
            try {
                Stop-Process -Id ([int]$PidVal) -Force -ErrorAction SilentlyContinue
            } catch {
                # Ignore if already stopped
            }
        }
    }
}

Write-Host ""
Write-Host "==============================================================================" -ForegroundColor Green
Write-Host "  [OK] All services have been stopped." -ForegroundColor Green
Write-Host "==============================================================================" -ForegroundColor Green
