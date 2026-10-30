# Register + run bot3 Ae-ft fold4 (sha 249b86fb) via schtasks (session-detached).
# ASCII-only. Copies runner + preflight .py into JobDir first.
# Weights expected at C:\Sloi\pc_jobs\weights\dfine_ae_ft_f4\ (from box dfine-ae-ft-f4pack).
$ErrorActionPreference = 'Continue'
$Task = 'koz3_bot3_dfine_aeor_ae_ft_f4'
$JobDir = 'C:\Sloi\pc_jobs\bot3_dfine_aeor_ae_ft_f4'
$SrcDir = Split-Path -Parent $MyInvocation.MyCommand.Path
New-Item -ItemType Directory -Force -Path $JobDir | Out-Null
Copy-Item -Force (Join-Path $SrcDir 'run_bot3_dfine_aeor_ae_ft_docker.ps1') "$JobDir\run_bot3_dfine_aeor_ae_ft_docker.ps1"
Copy-Item -Force (Join-Path $SrcDir 'bot3_ae_weights_preflight.py') "$JobDir\bot3_ae_weights_preflight.py"
$RunPs1 = "$JobDir\run_bot3_dfine_aeor_ae_ft_docker.ps1"
$Tr = "powershell.exe -NoProfile -ExecutionPolicy Bypass -File `"$RunPs1`" -Variant f4"
schtasks /Delete /TN $Task /F 2>$null | Out-Null
schtasks /Create /TN $Task /TR $Tr /SC ONCE /ST 00:00 /RL LIMITED /F
if ($LASTEXITCODE -ne 0) { Write-Host "schtasks Create failed rc=$LASTEXITCODE"; exit 1 }
schtasks /Run /TN $Task
Write-Host "launched $Task JobDir=$JobDir Variant=f4"
