Write-Host "Starting EchoCoach backend..." -ForegroundColor Cyan
Set-Location $PSScriptRoot
& "$PSScriptRoot\venv\Scripts\python.exe" -m uvicorn main:app --host 0.0.0.0 --port 8765 --reload
