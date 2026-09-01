$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

Write-Host "[1/2] Building latest V0.3 frontend..."
npm run build

Write-Host "[2/2] Packaging dist for server handoff..."
$zip = Join-Path $PSScriptRoot "materials-agent-v030-server-dist.zip"
if (Test-Path $zip) { Remove-Item $zip -Force }
Compress-Archive -Path (Join-Path $PSScriptRoot "dist") -DestinationPath $zip -Force

Write-Host "DONE: $zip"
