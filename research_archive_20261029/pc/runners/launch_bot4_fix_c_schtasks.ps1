# Register + run bot4 Fix C mock+honest via schtasks (session-detached). ASCII-only.
# Resolves tip from origin/bots/bot4-large-objects after fetch (override with KOZ3_SHA).
$ErrorActionPreference = 'Continue'
$Task = 'koz3_bot4_fix_c'
$JobDir = 'C:\Sloi\pc_jobs\bot4_fix_c'
$Repo = 'C:\Sloi\wt_bots'
$Branch = 'bots/bot4-large-objects'
$SrcDir = Split-Path -Parent $MyInvocation.MyCommand.Path
New-Item -ItemType Directory -Force -Path $JobDir | Out-Null
foreach ($f in @(
  'run_bot4_fix_c_docker.ps1',
  'bot3_ae_weights_preflight.py',
  'bot4_lo_ablate_summary.py'
)) {
  Copy-Item -Force (Join-Path $SrcDir $f) (Join-Path $JobDir $f)
}
& cmd.exe /c "git -C `"$Repo`" fetch origin $Branch" | Out-Null
if (-not $env:KOZ3_SHA) {
  $env:KOZ3_SHA = (& cmd.exe /c "git -C `"$Repo`" rev-parse --short origin/$Branch").Trim()
}
Write-Host "KOZ3_SHA=$($env:KOZ3_SHA)"
$RunPs1 = "$JobDir\run_bot4_fix_c_docker.ps1"
$Tr = "powershell.exe -NoProfile -ExecutionPolicy Bypass -File `"$RunPs1`""
schtasks /Delete /TN $Task /F 2>$null | Out-Null
schtasks /Create /TN $Task /TR $Tr /SC ONCE /ST 00:00 /RL LIMITED /F
if ($LASTEXITCODE -ne 0) { Write-Host "schtasks Create failed rc=$LASTEXITCODE"; exit 1 }
schtasks /Run /TN $Task
Write-Host "launched $Task JobDir=$JobDir sha=$($env:KOZ3_SHA)"
