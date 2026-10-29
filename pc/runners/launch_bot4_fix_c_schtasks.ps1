# Register + run bot4 Fix C mock+honest via schtasks (session-detached). ASCII-only.
$ErrorActionPreference = 'Continue'
$Task = 'koz3_bot4_fix_c'
$JobDir = 'C:\Sloi\pc_jobs\bot4_fix_c_17e35a5'
$SrcDir = Split-Path -Parent $MyInvocation.MyCommand.Path
New-Item -ItemType Directory -Force -Path $JobDir | Out-Null
foreach ($f in @(
  'run_bot4_fix_c_docker.ps1',
  'bot3_ae_weights_preflight.py',
  'bot4_lo_ablate_summary.py'
)) {
  Copy-Item -Force (Join-Path $SrcDir $f) (Join-Path $JobDir $f)
}
$RunPs1 = "$JobDir\run_bot4_fix_c_docker.ps1"
$Tr = "powershell.exe -NoProfile -ExecutionPolicy Bypass -File `"$RunPs1`""
schtasks /Delete /TN $Task /F 2>$null | Out-Null
schtasks /Create /TN $Task /TR $Tr /SC ONCE /ST 00:00 /RL LIMITED /F
if ($LASTEXITCODE -ne 0) { Write-Host "schtasks Create failed rc=$LASTEXITCODE"; exit 1 }
schtasks /Run /TN $Task
Write-Host "launched $Task JobDir=$JobDir"
