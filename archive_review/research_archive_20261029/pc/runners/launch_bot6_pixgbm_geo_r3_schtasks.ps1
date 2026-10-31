# Register + run bot6 PIXGBM geo r3 via schtasks (session-detached). ASCII-only.
$ErrorActionPreference = 'Continue'
$Task = 'koz3_bot6_pixgbm_geo_r3'
$JobDir = 'C:\Sloi\pc_jobs\bot6_pixgbm_geo_r3'
$SrcDir = Split-Path -Parent $MyInvocation.MyCommand.Path
New-Item -ItemType Directory -Force -Path $JobDir, "$JobDir\notes" | Out-Null
foreach ($f in @(
  'run_bot6_pixgbm_geo_r3.ps1',
  'bot6_r3_select_copy.py',
  'bot6_r3_build_gt.py',
  'bot6_r3_gate.py'
)) {
  Copy-Item -Force (Join-Path $SrcDir $f) (Join-Path $JobDir $f)
}
$RunPs1 = "$JobDir\run_bot6_pixgbm_geo_r3.ps1"
$Tr = "powershell.exe -NoProfile -ExecutionPolicy Bypass -File `"$RunPs1`""
schtasks /Delete /TN $Task /F 2>$null | Out-Null
schtasks /Create /TN $Task /TR $Tr /SC ONCE /ST 00:00 /RL LIMITED /F
if ($LASTEXITCODE -ne 0) { Write-Host "schtasks Create failed rc=$LASTEXITCODE"; exit 1 }
schtasks /Run /TN $Task
Write-Host "launched $Task JobDir=$JobDir"
