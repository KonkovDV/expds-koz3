# Register + run bot4 large_objects ablation via schtasks (session-detached).
# ASCII-only. Copies runner + helper .py into JobDir first.
$ErrorActionPreference = 'Continue'
$Task = 'koz3_bot4_lo_ablate'
$JobDir = 'C:\Sloi\pc_jobs\bot4_lo_ablate_5097c4a'
$SrcDir = Split-Path -Parent $MyInvocation.MyCommand.Path
New-Item -ItemType Directory -Force -Path $JobDir | Out-Null
Copy-Item -Force (Join-Path $SrcDir 'run_bot4_lo_ablate_docker.ps1') "$JobDir\run_bot4_lo_ablate_docker.ps1"
Copy-Item -Force (Join-Path $SrcDir 'bot4_lo_ablate_summary.py') "$JobDir\bot4_lo_ablate_summary.py"
Copy-Item -Force (Join-Path $SrcDir 'bot4_lo_ablate_torch_preflight.py') "$JobDir\bot4_lo_ablate_torch_preflight.py"
$RunPs1 = "$JobDir\run_bot4_lo_ablate_docker.ps1"
$Tr = "powershell.exe -NoProfile -ExecutionPolicy Bypass -File `"$RunPs1`""
schtasks /Delete /TN $Task /F 2>$null | Out-Null
schtasks /Create /TN $Task /TR $Tr /SC ONCE /ST 00:00 /RL LIMITED /F
if ($LASTEXITCODE -ne 0) { Write-Host "schtasks Create failed rc=$LASTEXITCODE"; exit 1 }
schtasks /Run /TN $Task
Write-Host "launched $Task JobDir=$JobDir"
