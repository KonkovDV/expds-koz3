# Register + run bot8 compliance pytest via schtasks (session-detached).
# ASCII-only. Copies runner into JobDir first.
# Set KOZ3_SUBMISSION to result.geojson (submission output) before launch if default missing.
$ErrorActionPreference = 'Continue'
$Task = 'koz3_bot8_compliance'
$JobDir = 'C:\Sloi\pc_jobs\bot8_compliance_f6fe20a'
$SrcDir = Split-Path -Parent $MyInvocation.MyCommand.Path
# Prefer JobDir matching runner SHA once pin known; fallback generic
if ($env:KOZ3_SHA) {
  $Short = $env:KOZ3_SHA.Substring(0,7)
  $JobDir = "C:\Sloi\pc_jobs\bot8_compliance_$Short"
}
New-Item -ItemType Directory -Force -Path $JobDir | Out-Null
Copy-Item -Force (Join-Path $SrcDir 'run_bot8_compliance.ps1') "$JobDir\run_bot8_compliance.ps1"
$RunPs1 = "$JobDir\run_bot8_compliance.ps1"
$Tr = "powershell.exe -NoProfile -ExecutionPolicy Bypass -File `"$RunPs1`""
schtasks /Delete /TN $Task /F 2>$null | Out-Null
schtasks /Create /TN $Task /TR $Tr /SC ONCE /ST 00:00 /RL LIMITED /F
if ($LASTEXITCODE -ne 0) { Write-Host "schtasks Create failed rc=$LASTEXITCODE"; exit 1 }
schtasks /Run /TN $Task
Write-Host "launched $Task JobDir=$JobDir (set KOZ3_SUBMISSION=path/to/result.geojson if needed)"
