$ErrorActionPreference = 'Continue'
[Console]::OutputEncoding = [Text.Encoding]::UTF8
$Job = 'bot3_aerial'
$Root = "C:\Sloi\pc_jobs\$Job"
$Wt = "C:\Sloi\wt_bot3_aerial"
$Py = 'C:\Sloi\expds-koz3_v0.2.0\expds-koz3\.venv\Scripts\python.exe'
$Train = 'F:\2\train-expds-3'
$Work = 'F:\expds3\work'
New-Item -ItemType Directory -Force -Path $Root | Out-Null
function Set-Status([string]$s) {
  $line = "$s $(Get-Date -Format o)"
  Set-Content -Encoding UTF8 "$Root\STATUS.txt" -Value $line
  Add-Content -Encoding UTF8 "$Root\heartbeat.log" -Value $line
}
Set-Status 'RUNNING'
$env:PYTHONPATH = $Wt
$env:PYTHONUNBUFFERED = '1'
$env:OMP_NUM_THREADS = '4'
Push-Location $Wt
try {
  & $Py -m pytest -q tests/test_bot3_ae_recall_relax.py tests/test_bot6_raster_fix.py tests/test_dfine_aeor.py --tb=line *> "$Root\pytest_A.log"
  if ($LASTEXITCODE -ne 0) { Set-Status 'FAILED pytest_A'; exit 1 }
  if (Test-Path "$Wt\scripts\eval_aerial_honest10.py") {
    & $Py scripts\eval_aerial_honest10.py --work $Work --train $Train --out "$Root\bot3_aerial_honest10.json" *> "$Root\honest_C.log"
    if ($LASTEXITCODE -ne 0) { Set-Status 'FAILED honest_C'; exit 2 }
  } else {
    'MISSING eval_aerial_honest10.py' | Set-Content "$Root\missing_C.txt"
  }
  Set-Status 'DONE'
} catch {
  $_ | Out-File -Encoding UTF8 "$Root\exception.txt"
  Set-Status 'FAILED exception'
  exit 9
} finally { Pop-Location }
