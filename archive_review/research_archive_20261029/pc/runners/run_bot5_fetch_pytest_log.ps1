# bot5: copy PC pytest.log @2b745df (+ env dump) into results. CPU ~1 min. No leases.
$ErrorActionPreference = 'Continue'
[Console]::OutputEncoding = [Text.Encoding]::UTF8

$Sha     = '2b745df167d315de3a377ca112d1283454ab1953'
$SrcDir  = 'C:\Sloi\pc_jobs\cpu_pytest_bot5_tip'
$JobDir  = 'C:\Sloi\pc_jobs\bot5_pytest_log_2b745df'
$Shared  = 'F:\Sloi_shared\koz3\pc\results\bot5_pytest_log_2b745df'  # optional mirror if F: shared mounted
$SharedAlt = @(
  'C:\Sloi\shared\koz3\pc\results\bot5_pytest_log_2b745df',
  'F:\Sloi_pc_jobs\bot5_pytest_log_2b745df'
)

New-Item -ItemType Directory -Force -Path $JobDir | Out-Null
function HB($m){ Add-Content -Encoding utf8 "$JobDir\heartbeat.log" "$(Get-Date -Format o) $m"; Write-Host $m }
function Set-Status($s){ $l="$s $(Get-Date -Format o)"; Set-Content -Encoding utf8 "$JobDir\STATUS.txt" $l; HB $l }

Set-Status 'RUNNING'
HB "fetch pytest log sha=$Sha from $SrcDir"

$log = Join-Path $SrcDir 'pytest.log'
if (-not (Test-Path $log)) {
  # try common alternates
  $alts = @(
    (Join-Path $SrcDir 'pytest.out'),
    (Join-Path $SrcDir 'out\pytest.log'),
    'C:\Sloi\pc_jobs\cpu_pytest_bot5_tip\pytest.log'
  )
  $found = $false
  foreach ($a in $alts) { if (Test-Path $a) { $log = $a; $found = $true; break } }
  if (-not $found) { Set-Status "FAILED missing_pytest_log under $SrcDir"; exit 2 }
}

Copy-Item -Force $log "$JobDir\pytest.log"
HB "copied pytest.log ($((Get-Item $log).Length) bytes)"

# Env dump from same host python used for that pytest if known; else PATH python
$PyCandidates = @(
  'C:\Sloi\expds-koz3_v0.2.0\expds-koz3\.venv\Scripts\python.exe',
  'C:\Sloi\wt_bots\.venv\Scripts\python.exe',
  (Get-Command python -EA SilentlyContinue | Select-Object -ExpandProperty Source)
) | Where-Object { $_ -and (Test-Path $_) }

$Py = $PyCandidates | Select-Object -First 1
if ($Py) {
  HB "python=$Py"
  & $Py -V *> "$JobDir\python_version.txt"
  & $Py -m pip freeze *> "$JobDir\pip_freeze.txt"
  Set-Content -Encoding utf8 "$JobDir\python_path.txt" $Py
} else {
  HB 'WARN no python found for freeze'
  Set-Content -Encoding utf8 "$JobDir\python_version.txt" 'NO_PYTHON_FOUND'
}

# Also copy any sibling logs if present
foreach ($extra in @('pytest.err','stdout.log','stderr.log','STATUS.txt')) {
  $p = Join-Path $SrcDir $extra
  if (Test-Path $p) { Copy-Item -Force $p "$JobDir\src_$extra" }
}

# Mirror to shared results paths if writable
$mirrors = @($Shared) + $SharedAlt
foreach ($dst in $mirrors) {
  try {
    $parent = Split-Path $dst -Parent
    if (Test-Path $parent) {
      New-Item -ItemType Directory -Force -Path $dst | Out-Null
      Copy-Item -Force "$JobDir\*" $dst -EA SilentlyContinue
      HB "mirrored -> $dst"
    }
  } catch { HB "mirror skip $dst : $_" }
}

# Always leave a copy under JobDir; operator/box sync via usual pc/results path
# If box shared is reachable as UNC or mapped drive used by other runners:
$BoxHint = 'pc/results/bot5_pytest_log_2b745df/'
Set-Content -Encoding utf8 "$JobDir\README.txt" @"
sha=$Sha
source=$SrcDir
copied=$(Get-Date -Format o)
Please sync this JobDir to box: $BoxHint
"@

Set-Status 'DONE'
HB 'DONE'
exit 0
