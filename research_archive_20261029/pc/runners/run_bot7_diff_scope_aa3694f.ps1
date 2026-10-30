# bot7 Δ decomposition dump @ aa3694f — CPU <=6 threads. Parallel OK with PIXGBM 12.
$ErrorActionPreference = 'Continue'
[Console]::OutputEncoding = [Text.Encoding]::UTF8
$JobDir = 'C:\Sloi\pc_jobs\bot7_diff_scope_aa3694f'
$Wt = 'C:\Sloi\wt_bot7_honest_layer'
$Sha = 'aa3694f'
$Full = 'aa3694f'  # resolve below
$Py = 'C:\Sloi\expds-koz3_v0.2.0\expds-koz3\.venv\Scripts\python.exe'
New-Item -ItemType Directory -Force -Path $JobDir, "$JobDir\notes" | Out-Null
function HB($m){ Add-Content -Encoding utf8 "$JobDir\heartbeat.log" "$(Get-Date -Format o) $m" }
function Set-Status($s){ $l="$s $(Get-Date -Format o)"; Set-Content -Encoding utf8 "$JobDir\STATUS.txt" $l; HB $l }

if ((Test-Path "$JobDir\STATUS.txt") -and ((Get-Content "$JobDir\STATUS.txt" -Raw) -match '^DONE')) { Write-Host 'already DONE'; exit 0 }
Set-Status 'RUNNING'

# Reset ONLY this wt
& cmd /c "git -C C:\Sloi\wt_bots fetch origin bots/bot7-honest-eval-layer" 2>&1 | Out-File "$JobDir\git_fetch.log"
# resolve full sha
$Full = (& cmd /c "git -C C:\Sloi\wt_bots rev-parse aa3694f").Trim()
HB "reset $Wt -> $Full"
& cmd /c "git -C `"$Wt`" fetch origin bots/bot7-honest-eval-layer" 2>&1 | Out-Null
& cmd /c "git -C `"$Wt`" checkout --detach $Full" 2>&1 | Out-File "$JobDir\git_co.log"
$head = (& cmd /c "git -C `"$Wt`" rev-parse --short HEAD").Trim()
HB "HEAD=$head"

$env:OMP_NUM_THREADS='6'; $env:OPENBLAS_NUM_THREADS='6'; $env:MKL_NUM_THREADS='6'; $env:PYTHONPATH=$Wt; $env:PYTHONUNBUFFERED='1'

Push-Location $Wt
try {
  & $Py -m pytest -q tests/test_bot7_diff_match_scope.py tests/test_bot7_layer_scope.py --tb=line *> "$JobDir\pytest.log"
  HB "pytest exit=$LASTEXITCODE"
  if ($LASTEXITCODE -ne 0) { Set-Status 'FAILED pytest'; exit 1 }

  $t1=Get-Date
  $p=Start-Process -FilePath $Py -ArgumentList @(
    'scripts\diff_match_scope.py',
    '--work','F:\expds3\work',
    '--train','F:\2\train-expds-3',
    '--metric','materials\compute_metrics.py',
    '--honest','10',
    '--out', $JobDir
  ) -WorkingDirectory $Wt -PassThru -NoNewWindow -RedirectStandardOutput "$JobDir\diff.out" -RedirectStandardError "$JobDir\diff.err"
  while(-not $p.HasExited){ HB "diff pid=$($p.Id) $([int]((Get-Date)-$t1).TotalSeconds)s"; Start-Sleep 20 }
  $sec=[math]::Round(((Get-Date)-$t1).TotalSeconds,1)
  Set-Content "$JobDir\wall_sec.txt" $sec
  HB "diff exit=$($p.ExitCode) sec=$sec"
  if (-not (Test-Path "$JobDir\summary.json")) { Set-Status 'FAILED no summary'; exit 2 }
  Set-Status 'DONE'
} catch {
  $_ | Out-File "$JobDir\exception.txt"
  Set-Status 'FAILED exception'
  exit 9
} finally { Pop-Location }
