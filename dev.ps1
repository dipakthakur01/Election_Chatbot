Param([string]$Port = "8000", [switch]$Install)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $MyInvocation.MyCommand.Path)
if (!(Test-Path ".\.venv")) { python -m venv .venv }
. .\.venv\Scripts\Activate.ps1
if ($Install) {
    pip install -r requirements.txt
}
$Url = "http://localhost:$Port/"
Write-Host $Url
Start-Process $Url
python -m uvicorn app.main:app --reload --port $Port
