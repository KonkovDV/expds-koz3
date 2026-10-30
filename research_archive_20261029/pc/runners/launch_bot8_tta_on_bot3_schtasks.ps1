# Register + run bot8 TTA-on-bot3 Docker gate via schtasks. ASCII-only.
# AFTER first bot8 gate (bots/bot8-tta-lb @ f6fe20a).
$ErrorActionPreference = 'Continue'
$Task = 'koz3_bot8_tta_on_bot3'
$JobDir = 'C:\Sloi\pc_jobs\bot8_tta_on_bot3_pending'
$SrcDir = Split-Path -Parent $MyInvocation.MyCommand.Path
if ($env:KOZ3_SHA) {
  $Short = $env:KOZ3_SHA.Substring(0,7)
  $JobDir = "C:\Sloi\pc_jobs\bot8_tta_on_bot3_$Short"
}
New-Item -ItemType Directory -Force -Path $JobDir | Out-Null
Copy-Item -Force (Join-Path $SrcDir 'run_bot8_tta_on_bot3_docker.ps1') "$JobDir\run_bot8_tta_on_bot3_docker.ps1"
if (Test-Path (Join-Path $SrcDir 'bot3_ae_weights_preflight.py')) {
  Copy-Item -Force (Join-Path $SrcDir 'bot3_ae_weights_preflight.py') "$JobDir\bot3_ae_weights_preflight.py"
}
$RunPs1 = "$JobDir\run_bot8_tta_on_bot3_docker.ps1"
$Tr = "powershell.exe -NoProfile -ExecutionPolicy Bypass -File `"$RunPs1`""
schtasks /Delete /TN $Task /F 2>$null | Out-Null
schtasks /Create /TN $Task /TR $Tr /SC ONCE /ST 00:00 /RL LIMITED /F
if ($LASTEXITCODE -ne 0) { Write-Host "schtasks Create failed rc=$LASTEXITCODE"; exit 1 }
schtasks /Run /TN $Task
Write-Host "launched $Task JobDir=$JobDir"
