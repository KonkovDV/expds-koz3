# Register + run bot8 TTA/MS Docker gate via schtasks (session-detached).
# ASCII-only. Copies runner into JobDir first. Image pantum/n7q:20261024.
$ErrorActionPreference = 'Continue'
$Task = 'koz3_bot8_tta_lb'
$JobDir = 'C:\Sloi\pc_jobs\bot8_tta_lb_docker_f6fe20a'
$SrcDir = Split-Path -Parent $MyInvocation.MyCommand.Path
if ($env:KOZ3_SHA) {
  $Short = $env:KOZ3_SHA.Substring(0,7)
  $JobDir = "C:\Sloi\pc_jobs\bot8_tta_lb_docker_$Short"
}
New-Item -ItemType Directory -Force -Path $JobDir | Out-Null
Copy-Item -Force (Join-Path $SrcDir 'run_bot8_tta_ms_docker.ps1') "$JobDir\run_bot8_tta_ms_docker.ps1"
$RunPs1 = "$JobDir\run_bot8_tta_ms_docker.ps1"
$Tr = "powershell.exe -NoProfile -ExecutionPolicy Bypass -File `"$RunPs1`""
schtasks /Delete /TN $Task /F 2>$null | Out-Null
schtasks /Create /TN $Task /TR $Tr /SC ONCE /ST 00:00 /RL LIMITED /F
if ($LASTEXITCODE -ne 0) { Write-Host "schtasks Create failed rc=$LASTEXITCODE"; exit 1 }
schtasks /Run /TN $Task
Write-Host "launched $Task JobDir=$JobDir"
