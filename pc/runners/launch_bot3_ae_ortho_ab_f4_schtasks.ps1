# Launch Fix A+B Docker gate (f4pack) via schtasks — ASCII only.
$ErrorActionPreference = 'Stop'
$SrcDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Task = 'koz3_bot3_ae_ortho_ab_f4'
$JobDir = 'C:\Sloi\pc_jobs\bot3_ae_ortho_ab_f4'
New-Item -ItemType Directory -Force -Path $JobDir, "$JobDir\notes" | Out-Null
Copy-Item -Force (Join-Path $SrcDir 'run_bot3_ae_ortho_ab_docker.ps1') "$JobDir\run_bot3_ae_ortho_ab_docker.ps1"
Copy-Item -Force (Join-Path $SrcDir 'bot3_ae_weights_preflight.py') "$JobDir\bot3_ae_weights_preflight.py" -EA SilentlyContinue
$RunPs1 = "$JobDir\run_bot3_ae_ortho_ab_docker.ps1"
$Arg = "-NoProfile -ExecutionPolicy Bypass -File `"$RunPs1`" -Variant f4"
schtasks /Delete /TN $Task /F 2>$null | Out-Null
schtasks /Create /TN $Task /SC ONCE /ST 00:00 /RL HIGHEST /RU SYSTEM /TR "powershell.exe $Arg" /F | Out-Null
schtasks /Run /TN $Task | Out-Null
Write-Output "launched $Task JobDir=$JobDir"
