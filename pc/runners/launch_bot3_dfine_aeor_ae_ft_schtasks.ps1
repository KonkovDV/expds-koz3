# Register + run bot3 Ae-ft final (sha 3e619878) via schtasks (session-detached).
# ASCII-only. Copies runner + preflight .py into JobDir first.
$ErrorActionPreference = 'Continue'
$Task = 'koz3_bot3_dfine_aeor_ae_ft'
$JobDir = 'C:\Sloi\pc_jobs\bot3_dfine_aeor_ae_ft'
$SrcDir = Split-Path -Parent $MyInvocation.MyCommand.Path
New-Item -ItemType Directory -Force -Path $JobDir | Out-Null
Copy-Item -Force (Join-Path $SrcDir 'run_bot3_dfine_aeor_ae_ft_docker.ps1') "$JobDir\run_bot3_dfine_aeor_ae_ft_docker.ps1"
Copy-Item -Force (Join-Path $SrcDir 'bot3_ae_weights_preflight.py') "$JobDir\bot3_ae_weights_preflight.py"
$RunPs1 = "$JobDir\run_bot3_dfine_aeor_ae_ft_docker.ps1"
$Tr = "powershell.exe -NoProfile -ExecutionPolicy Bypass -File `"$RunPs1`" -Variant final"
schtasks /Delete /TN $Task /F 2>$null | Out-Null
schtasks /Create /TN $Task /TR $Tr /SC ONCE /ST 00:00 /RL LIMITED /F
if ($LASTEXITCODE -ne 0) { Write-Host "schtasks Create failed rc=$LASTEXITCODE"; exit 1 }
schtasks /Run /TN $Task
Write-Host "launched $Task JobDir=$JobDir Variant=final"
