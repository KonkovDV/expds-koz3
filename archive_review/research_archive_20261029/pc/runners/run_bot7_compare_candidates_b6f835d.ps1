# bot7 compare_candidates scoring @ b6f835d (CPU <=4, no inference)
$ErrorActionPreference = 'Continue'
[Console]::OutputEncoding = [Text.Encoding]::UTF8

$Sha     = 'b6f835dc8bb58e238dba74ec8a9ecad4cd52c434'
$JobDir  = 'F:\Sloi_pc_jobs\bot7_compare_candidates_b6f835d'
$Wt      = 'F:\Sloi_wt\wt_bot7_compare_candidates'
$Image   = 'pantum/n7q:20261024'
$Threads = '4'

$GT      = 'F:\Sloi_pc_jobs_archive\v3e2e\out_v3fix2\gt.geojson'
$V2      = 'F:\Sloi_pc_jobs_archive\v3e2e\out_v3fix2\res_v2'
$V3      = 'F:\Sloi_pc_jobs_archive\v3e2e\out_v3fix2\res_v3'
$HOTFIX  = 'F:\Sloi_pc_jobs_archive\bot2_v3_hotfix_1ec2d7c\out_B'
$V4OFF   = 'C:\Sloi\pc_jobs\bot5_v4_off_50f6f27'
$B4OFF   = 'F:\Sloi_pc_jobs_archive\bot4_lo_mock_337a27c\out_off'
$B4ON    = 'F:\Sloi_pc_jobs_archive\bot4_lo_mock_337a27c\out_on'
$FallbackGT = 'F:\expds3\mock_val'

New-Item -ItemType Directory -Force -Path $JobDir, "$JobDir\notes", "$JobDir\compare" | Out-Null

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

$gtUsed = $GT
if (-not (Test-Path $GT)) {
  Heartbeat "GT missing at $GT - fallback $FallbackGT"
  $gtUsed = $FallbackGT
}
if (-not (Test-Path $gtUsed)) { Set-Status "FAILED missing_gt $gtUsed"; exit 3 }
$gtFile = if ((Get-Item $gtUsed).PSIsContainer) { Join-Path $gtUsed 'gt.geojson' } else { $gtUsed }
$gtHash = (Get-FileHash -Algorithm SHA256 -Path $gtFile).Hash.ToLower()
Heartbeat "GT=$gtFile sha256=$gtHash"
Set-Content -Encoding utf8 "$JobDir\notes\input_hashes.txt" -Value "GT $gtHash $gtFile"

if (-not (Test-Path "$Wt\.git")) {
  Heartbeat 'worktree add'
  & cmd.exe /c "git -C C:\Sloi\wt_bots fetch origin bots/bot7-compare-candidates" 2>&1 | Out-File -Encoding utf8 "$JobDir\notes\git_fetch.log"
  & cmd.exe /c "git -C C:\Sloi\wt_bots worktree add --detach `"$Wt`" $Sha" 2>&1 | Out-File -Encoding utf8 "$JobDir\notes\git_wt.log"
} else {
  & cmd.exe /c "git -C `"$Wt`" fetch origin bots/bot7-compare-candidates" 2>&1 | Out-File -Encoding utf8 "$JobDir\notes\git_fetch.log"
  & cmd.exe /c "git -C `"$Wt`" checkout --detach $Sha" 2>&1 | Out-File -Encoding utf8 "$JobDir\notes\git_wt.log"
}
$head = (& cmd.exe /c "git -C `"$Wt`" rev-parse HEAD").Trim()
Heartbeat "worktree HEAD=$head"
if ($head.Substring(0,7) -ne $Sha.Substring(0,7)) { Set-Status "FAILED bad_sha want=$Sha got=$head"; exit 2 }

$labels = @('v2','v3','hotfix_B','v4_off','bot4_off','bot4_on')
$cands = @($V2,$V3,$HOTFIX,$V4OFF,$B4OFF,$B4ON)
for ($i=0; $i -lt $cands.Count; $i++) {
  $p = $cands[$i]; $lab = $labels[$i]
  if (-not (Test-Path $p)) { Set-Status "FAILED missing_cand $lab=$p"; exit 3 }
  $g = $null
  if (Test-Path (Join-Path $p 'result.geojson')) { $g = Join-Path $p 'result.geojson' }
  elseif (Test-Path (Join-Path $p 'out\result.geojson')) { $g = Join-Path $p 'out\result.geojson' }
  else {
    $hit = Get-ChildItem $p -Recurse -Filter 'result.geojson' -EA SilentlyContinue | Select-Object -First 1
    if ($hit) { $g = $hit.FullName }
  }
  if ($g) {
    $h = (Get-FileHash -Algorithm SHA256 -Path $g).Hash.ToLower()
    Add-Content -Encoding utf8 "$JobDir\notes\input_hashes.txt" -Value "$lab $h $g"
    Heartbeat "cand $lab -> $g sha256=$h"
  } else {
    Heartbeat "cand $lab folder=$p (find_geojson)"
  }
}

$useDocker = [bool](docker images -q $Image 2>$null)
$rc = 1
$envUsed = 'unknown'

if ($useDocker) {
  $CName = 'koz3-bot7-compare-' + (Get-Date -Format 'yyyyMMddHHmmss')
  $JobM = ($JobDir -replace '\\','/')
  $WtM  = ($Wt -replace '\\','/')
  $sha7 = $Sha.Substring(0,7)
  $innerLines = @(
    'set -e',
    "export OMP_NUM_THREADS=$Threads PYTHONPATH=/repo",
    'cd /repo',
    ('python scripts/compare_candidates.py --gt /data/gt.geojson --v2 /data/res_v2 --out /work/compare --workers {0} --labels v2,v3,hotfix_B,v4_off,bot4_off,bot4_on --branch bots/bot7-compare-candidates --sha {1} --experiments-row --experiments-file /work/compare/EXPERIMENTS_rows.csv /data/res_v2 /data/res_v3 /data/hotfix_B /data/v4_off /data/bot4_off /data/bot4_on' -f $Threads, $sha7)
  )
  $innerPath = Join-Path $JobDir 'notes\inner.sh'
  [IO.File]::WriteAllText($innerPath, (($innerLines -join "`n") + "`n"), [Text.UTF8Encoding]::new($false))

  Heartbeat "docker run compare name=$CName"
  docker run --rm --name $CName --network none `
    -e OMP_NUM_THREADS=$Threads `
    -v "${WtM}:/repo:ro" `
    -v "${JobM}:/work" `
    -v "F:/Sloi_pc_jobs_archive/v3e2e/out_v3fix2/gt.geojson:/data/gt.geojson:ro" `
    -v "F:/Sloi_pc_jobs_archive/v3e2e/out_v3fix2/res_v2:/data/res_v2:ro" `
    -v "F:/Sloi_pc_jobs_archive/v3e2e/out_v3fix2/res_v3:/data/res_v3:ro" `
    -v "F:/Sloi_pc_jobs_archive/bot2_v3_hotfix_1ec2d7c/out_B:/data/hotfix_B:ro" `
    -v "C:/Sloi/pc_jobs/bot5_v4_off_50f6f27:/data/v4_off:ro" `
    -v "F:/Sloi_pc_jobs_archive/bot4_lo_mock_337a27c/out_off:/data/bot4_off:ro" `
    -v "F:/Sloi_pc_jobs_archive/bot4_lo_mock_337a27c/out_on:/data/bot4_on:ro" `
    -w /repo $Image bash /work/notes/inner.sh 2>&1 |
      Tee-Object -FilePath "$JobDir\compare.docker.log"
  $rc = $LASTEXITCODE
  $envUsed = "docker:$Image"
  Heartbeat "docker exit rc=$rc"
}

if ($rc -ne 0) {
  Heartbeat 'docker failed or skipped - trying host python'
  $py = $null
  foreach ($c in @(
    'C:\Sloi\wt_bots\.venv\Scripts\python.exe',
    'C:\Sloi\pc_jobs\_venv\Scripts\python.exe',
    'C:\Sloi\.venv\Scripts\python.exe'
  )) { if (Test-Path $c) { $py = $c; break } }
  if (-not $py) {
    $py = (Get-Command python -EA SilentlyContinue | Select-Object -ExpandProperty Source)
  }
  if (-not $py) { Set-Status 'FAILED no_python'; exit 4 }
  Heartbeat "host py=$py"
  $env:OMP_NUM_THREADS = $Threads
  $env:PYTHONPATH = $Wt
  Push-Location $Wt
  & $py scripts/compare_candidates.py `
    --gt $gtFile --v2 $V2 --out (Join-Path $JobDir 'compare') --workers ([int]$Threads) `
    --labels v2,v3,hotfix_B,v4_off,bot4_off,bot4_on `
    --branch bots/bot7-compare-candidates --sha $Sha.Substring(0,7) `
    --experiments-row --experiments-file (Join-Path $JobDir 'compare\EXPERIMENTS_rows.csv') `
    $V2 $V3 $HOTFIX $V4OFF $B4OFF $B4ON 2>&1 |
      Tee-Object -FilePath "$JobDir\compare.host.log"
  $rc = $LASTEXITCODE
  Pop-Location
  $envUsed = "host:$py"
}

Add-Content -Encoding utf8 "$JobDir\STATUS.txt" -Value "ENV=$envUsed"
Heartbeat "ENV=$envUsed"

if ($rc -ne 0) { Set-Status "FAILED compare_rc=$rc ENV=$envUsed"; exit $rc }
if (-not (Test-Path "$JobDir\compare\compare.md")) { Set-Status "FAILED no_compare_md ENV=$envUsed"; exit 5 }

Set-Status "DONE ENV=$envUsed"
exit 0
