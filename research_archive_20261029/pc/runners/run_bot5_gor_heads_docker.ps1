# bot5 gor_v3 + gor_dense head retrain in Docker (pantum/n7q).
# Host Windows .venv cannot load inference/lib/vendor (manylinux) / transformers.
# Train MUST run in Linux container with vendor from v3on zip.
$ErrorActionPreference = 'Continue'
[Console]::OutputEncoding = [Text.Encoding]::UTF8

$Sha     = '9f62a9676b84e6963b892e6eb9f0907ad975f6ba'
$JobDir  = 'F:\Sloi_pc_jobs\bot5_gor_heads_9f62a96'
$Wt      = 'F:\Sloi_wt\wt_bot5_gor_heads'
$Image   = 'pantum/n7q:20261024'
$Data    = 'F:\2\train-expds-3'
$Model   = 'C:\Sloi\wt_bot5_50f6f27\models\dinov2-small'
$ZipV3   = 'C:\Sloi\pc_jobs\_vendor_zips\submission_bot7cand_v3on_0876090.zip'
$Threads = '8'
$Device  = 'cpu'

New-Item -ItemType Directory -Force -Path $JobDir, "$JobDir\notes" | Out-Null

function Set-Status([string]$s) {
  $line = "$s $(Get-Date -Format o)"
  Set-Content -Encoding utf8 "$JobDir\STATUS.txt" -Value $line
  Add-Content -Encoding utf8 "$JobDir\heartbeat.log" -Value $line
  Write-Host $line
}
function Heartbeat([string]$msg) {
  Add-Content -Encoding utf8 "$JobDir\heartbeat.log" -Value "$(Get-Date -Format o) $msg"
  Write-Host $msg
}

$st = if (Test-Path "$JobDir\STATUS.txt") { Get-Content "$JobDir\STATUS.txt" -Raw } else { '' }
if ($st -match '^DONE') { Write-Host 'already DONE'; exit 0 }

Set-Status 'RUNNING'
Heartbeat "start sha=$Sha image=$Image device=$Device threads=$Threads"

# --- worktree ---
if (-not (Test-Path "$Wt\.git")) {
  Heartbeat 'worktree add'
  & cmd.exe /c "git -C C:\Sloi\wt_bots fetch origin bots/bot5-candidate-v4" 2>&1 |
    Out-File -Encoding utf8 "$JobDir\notes\git_fetch.log"
  & cmd.exe /c "git -C C:\Sloi\wt_bots worktree add --detach `"$Wt`" $Sha" 2>&1 |
    Out-File -Encoding utf8 "$JobDir\notes\git_wt.log"
} else {
  & cmd.exe /c "git -C `"$Wt`" fetch origin bots/bot5-candidate-v4" 2>&1 |
    Out-File -Encoding utf8 "$JobDir\notes\git_fetch.log"
  & cmd.exe /c "git -C `"$Wt`" checkout --detach $Sha" 2>&1 |
    Out-File -Encoding utf8 "$JobDir\notes\git_wt.log"
}
$head = (& cmd.exe /c "git -C `"$Wt`" rev-parse HEAD").Trim()
Heartbeat "worktree HEAD=$head"
if ($head.Substring(0, [Math]::Min(7, $head.Length)) -ne $Sha.Substring(0, [Math]::Min(7, $Sha.Length)) -and $head -ne $Sha) {
  Set-Status "FAILED bad_sha want=$Sha got=$head"
  exit 2
}

# --- ensure vendor (manylinux) from v3on zip ---
$Vendor = Join-Path $Wt 'inference\lib\vendor\transformers'
if (-not (Test-Path $Vendor)) {
  Heartbeat "vendor missing - unpacking from $ZipV3"
  if (-not (Test-Path $ZipV3)) {
    Set-Status "FAILED missing_vendor_zip $ZipV3"
    exit 3
  }
  $Tmp = Join-Path $JobDir 'vendor_unpack'
  New-Item -ItemType Directory -Force -Path $Tmp | Out-Null
  & tar -xf $ZipV3 -C $Tmp --wildcards 'inference/lib/vendor/*' 2>&1 |
    Out-File -Encoding utf8 "$JobDir\notes\vendor_tar.log"
  if (-not (Test-Path (Join-Path $Tmp 'inference\lib\vendor\transformers'))) {
    Heartbeat 'tar wildcards failed - Expand-Archive full zip (slow)'
    Expand-Archive -Path $ZipV3 -DestinationPath $Tmp -Force
  }
  $SrcVendor = Join-Path $Tmp 'inference\lib\vendor'
  if (-not (Test-Path (Join-Path $SrcVendor 'transformers'))) {
    Set-Status "FAILED vendor_not_in_zip $ZipV3"
    exit 3
  }
  New-Item -ItemType Directory -Force -Path (Join-Path $Wt 'inference\lib') | Out-Null
  Copy-Item -Recurse -Force $SrcVendor (Join-Path $Wt 'inference\lib\vendor')
  Heartbeat 'vendor copied into worktree'
} else {
  Heartbeat 'vendor already present'
}

# --- preflight + train inside Docker ---
$WorkHost = $JobDir
$DataM  = ($Data -replace '\\', '/')
$ModelM = ($Model -replace '\\', '/')
$WtM    = ($Wt -replace '\\', '/')
$WorkM  = ($WorkHost -replace '\\', '/')

$GpuArgs = @()
if ($Device -eq 'cuda') { $GpuArgs = @('--gpus', 'all') }

# Write inner.sh without a PS here-string (avoids quote/encoding traps)
$InnerLines = @(
  'set -e',
  "export REPO=/repo DATA=/data MODEL=/model WORK=/work DEVICE=$Device THREADS=$Threads OMP_NUM_THREADS=$Threads PYTHONPATH=/repo",
  'export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1',
  'cd /repo',
  "python -c ""from scripts.gor_heads.dinov2_loader import probe_transformers, preflight_model_dir; from pathlib import Path; r=preflight_model_dir(Path('/model')); print('model_preflight', r); i=probe_transformers(Path('/repo')); print('tr', i.get('source'), i.get('version'), i.get('vendor_present'))""",
  'python scripts/gor_heads/train_heads.py'
)
$InnerPath = Join-Path $JobDir 'notes\inner.sh'
# LF line endings for bash
[IO.File]::WriteAllText($InnerPath, (($InnerLines -join "`n") + "`n"), [Text.UTF8Encoding]::new($false))

$CName = 'koz3-bot5-gor-heads-' + (Get-Date -Format 'yyyyMMddHHmmss')
Heartbeat "docker run train_heads name=$CName"
$cmd = @(
  'docker', 'run', '--rm'
) + $GpuArgs + @(
  '--name', $CName,
  '--network', 'none',
  '--shm-size', '8g',
  '-e', "DEVICE=$Device",
  '-e', "THREADS=$Threads",
  '-v', "${WtM}:/repo",
  '-v', "${DataM}:/data:ro",
  '-v', "${ModelM}:/model:ro",
  '-v', "${WorkM}:/work",
  '-w', '/repo',
  $Image,
  'bash', '/work/notes/inner.sh'
)
& $cmd[0] $cmd[1..($cmd.Length-1)] 2>&1 | Tee-Object -FilePath "$JobDir\train_heads.docker.log"
$rc = $LASTEXITCODE
Heartbeat "docker exit rc=$rc"

if (Test-Path "$JobDir\STATUS.txt") {
  $body = Get-Content "$JobDir\STATUS.txt" -Raw
  if ($body -match '^DONE') {
    Heartbeat 'DONE from train_heads STATUS'
    exit 0
  }
  if ($body -match '^FAILED') {
    Heartbeat 'FAILED from train_heads STATUS'
    exit 1
  }
}
if ($rc -ne 0) {
  Set-Status "FAILED docker_rc=$rc"
  exit $rc
}
Set-Status 'FAILED no_DONE_status'
exit 1
