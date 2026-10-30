# Detached-safe bot7 honest-10 global vs layer. PC job runner.
$ErrorActionPreference = 'Continue'
[Console]::OutputEncoding = [Text.Encoding]::UTF8
$Job = 'bot7_honest_layer'
$Root = "C:\Sloi\pc_jobs\$Job"
$Wt = "C:\Sloi\wt_bot7_honest_layer"
$Py = 'C:\Sloi\expds-koz3_v0.2.0\expds-koz3\.venv\Scripts\python.exe'
$Train = 'F:\2\train-expds-3'
$Work = 'F:\expds3\work'
New-Item -ItemType Directory -Force -Path $Root, "$Root\notes" | Out-Null
function Set-Status([string]$s) {
  $line = "$s $(Get-Date -Format o)"
  Set-Content -Encoding UTF8 "$Root\STATUS.txt" -Value $line
  Add-Content -Encoding UTF8 "$Root\heartbeat.log" -Value $line
}
Set-Status 'RUNNING'
$env:PYTHONPATH = $Wt
$env:PYTHONUNBUFFERED = '1'
$env:OMP_NUM_THREADS = '4'
$env:OPENBLAS_NUM_THREADS = '4'
$env:MKL_NUM_THREADS = '4'
Push-Location $Wt
try {
  & $Py -m pytest -q tests/test_bot7_layer_scope.py tests/test_bot8_scope_default.py tests/test_bot8_site_scope.py -k "not real_neighbour and not site_fast_collect" --tb=line *> "$Root\pytest.log"
  if ($LASTEXITCODE -ne 0) { Set-Status 'FAILED pytest'; exit 1 }
  # ADAPTED: train path F:\2\train-expds-3 (not F:\expds3\train-expds-3); metric from worktree
  $metric = Join-Path $Wt 'materials\compute_metrics.py'
  & $Py scripts\scoreboard.py --work $Work --train $Train --metric $metric --honest 10 --out "$Root\notes\bot7_honest_global_ae4c816.json" *> "$Root\scoreboard_global.log"
  if ($LASTEXITCODE -ne 0) { Set-Status 'FAILED scoreboard_global'; exit 2 }
  & $Py scripts\scoreboard.py --work $Work --train $Train --metric $metric --honest 10 --match-scope layer --out "$Root\notes\bot7_honest_layer_ae4c816.json" *> "$Root\scoreboard_layer.log"
  if ($LASTEXITCODE -ne 0) { Set-Status 'FAILED scoreboard_layer'; exit 3 }
  Set-Status 'DONE'
} catch {
  $_ | Out-File -Encoding UTF8 "$Root\exception.txt"
  Set-Status 'FAILED exception'
  exit 9
} finally { Pop-Location }
