# bot2 v2 gate baseline in Docker (pantum/n7q) — NOT host venv. Modeled on run_bot2_a3_dfine_docker_import.ps1.
# Zip = submission_bot7cand_v2_eb268d6 (sp_dfine OFF). Produces mock ALL/layers + optional honest-10.
# Container name prefix koz3- only; never touch talent/kontur/microphoenix.
$ErrorActionPreference = 'Continue'
[Console]::OutputEncoding = [Text.Encoding]::UTF8
$JobDir = 'C:\Sloi\pc_jobs\bot2_v2_base_docker_eb268d6'
$ShaZip = 'eb268d6e4adfbed32670cabeef15a709649b8246'
$PyHost = 'C:\Sloi\expds-koz3_v0.2.0\expds-koz3\.venv\Scripts\python.exe'  # scoring only
$Image = 'pantum/n7q:20261024'
$Regions = 'F:\Sloi_pc_jobs_archive\v3e2e\regions'
$ZipV2 = 'F:\Sloi_pc_jobs_archive\v3e2e\kit\submission_bot7cand_v2_eb268d6.zip'
$ToolsTar = 'F:\Sloi_pc_jobs_archive\v3e2e\kit\tools_v3e2e.tar'
$RefV2 = 'F:\Sloi_pc_jobs_archive\v3e2e\out_v3fix2\res_v2\result.geojson'
if (-not (Test-Path $RefV2)) { $RefV2 = 'F:\Sloi_pc_jobs_archive\v3e2e\out\res_v2\result.geojson' }
$Name = 'koz3-bot2-v2base-' + (Get-Date -Format 'yyyyMMddHHmmss')
New-Item -ItemType Directory -Force -Path $JobDir, "$JobDir\out_mock", "$JobDir\notes", "$JobDir\zip_work" | Out-Null

function HB($m){ Add-Content -Encoding utf8 "$JobDir\heartbeat.log" "$(Get-Date -Format o) $m" }
function Set-Status($s){ $l="$s $(Get-Date -Format o)"; Set-Content -Encoding utf8 "$JobDir\STATUS.txt" $l; HB $l }

$st = if (Test-Path "$JobDir\STATUS.txt") { Get-Content "$JobDir\STATUS.txt" -Raw } else { '' }
if ($st -match '^DONE') { Write-Host 'already DONE'; exit 0 }

Set-Status 'RUNNING'
HB "start v2 base docker name=$Name sha_zip=$ShaZip"

if (-not (Test-Path $ZipV2)) { Set-Status "FAILED no_zip $ZipV2"; exit 2 }

# unpack zip once (no overlay — pure v2)
$Z = "$JobDir\zip_work"
if (-not (Test-Path "$Z\inference\main.py")) {
  HB 'expand zip v2'
  Expand-Archive -Force -Path $ZipV2 -DestinationPath $Z
}
# confirm sp_dfine off
$toml = Get-ChildItem -Path $Z -Recurse -Filter 'default.toml' -EA SilentlyContinue | Select-Object -First 1 -ExpandProperty FullName
if ($toml) {
  Select-String -Path $toml -Pattern '\[sp_dfine\]|enabled =|\[gor_gate\]' | Select-Object -First 20 |
    ForEach-Object { $_.Line } | Set-Content -Encoding UTF8 "$JobDir\toml_snip.txt"
}

# tools for score
if (-not (Test-Path "$JobDir\tools")) {
  New-Item -ItemType Directory -Force -Path "$JobDir\tools" | Out-Null
  tar -xf $ToolsTar -C $JobDir 2>$null
}
$ScorePy = Get-ChildItem -Path $JobDir -Recurse -Filter e2e_score.py -EA SilentlyContinue | Select-Object -First 1 -ExpandProperty FullName
$MetricPy = Get-ChildItem -Path $JobDir -Recurse -Filter compute_metrics.py -EA SilentlyContinue | Select-Object -First 1 -ExpandProperty FullName
HB "scorer=$ScorePy"

HB 'docker ps before'
docker ps -a --format '{{.Names}} {{.Status}}' 2>&1 | Tee-Object "$JobDir\docker_ps_before.txt" | Out-Null

$ZMount = ($Z -replace '\\','/')
$RegMount = ($Regions -replace '\\','/')
$OutMount = ($JobDir -replace '\\','/')
# wipe stale geojson so salvage cannot accept pre-run file (LESSONS §2 / A4 bug)
New-Item -ItemType Directory -Force -Path "$JobDir\out_mock" | Out-Null
Remove-Item "$JobDir\out_mock\result.geojson" -Force -EA SilentlyContinue
HB "docker run $Name"
$t1 = Get-Date
$dockerArgs = @(
  'run','--name', $Name,
  '--gpus','all',
  '--network','none',
  '-e', 'PYTHONDONTWRITEBYTECODE=1',
  '-e', 'PYTHONUNBUFFERED=1',
  '-v', "${ZMount}:/data/zip:ro",
  '-v', "${RegMount}:/data/regions:ro",
  '-v', "${OutMount}:/data/output",
  '-w', '/data/zip',
  $Image,
  'python', 'inference/main.py', '/data/regions', '/data/output/out_mock/result.geojson'
)
$p = Start-Process -FilePath 'docker' -ArgumentList $dockerArgs -PassThru -NoNewWindow `
  -RedirectStandardOutput "$JobDir\main_mock.log" -RedirectStandardError "$JobDir\main_mock.err"
while (-not $p.HasExited) {
  HB "docker pid=$($p.Id) $([int]((Get-Date)-$t1).TotalSeconds)s"
  Start-Sleep 20
}
$sec = [math]::Round(((Get-Date)-$t1).TotalSeconds,1)
$ec = $p.ExitCode; if ($null -eq $ec) { $ec = -1 }
HB "docker exit=$ec sec=$sec"
Set-Content "$JobDir\wall_sec.txt" $sec
docker rm -f $Name 2>&1 | Tee-Object "$JobDir\docker_rm.txt" | Out-Null

# warn scan
Select-String -Path "$JobDir\main_mock.log","$JobDir\main_mock.err" -Pattern 'SP_DFINE_UNAVAILABLE|UNAVAILABLE|ускоритель|среда:|platform' -EA SilentlyContinue |
  ForEach-Object { $_.Line } | Set-Content -Encoding UTF8 "$JobDir\warn_scan.txt"

# score mock
$Gt = "$JobDir\gt.geojson"
if ($ScorePy -and (Test-Path "$JobDir\out_mock\result.geojson")) {
  if (-not (Test-Path $Gt)) {
    # prefer kit gt if present
    $KitGt = 'F:\Sloi_pc_jobs_archive\v3e2e\out_v3fix2\gt.geojson'
    if (Test-Path $KitGt) { Copy-Item $KitGt $Gt } else {
      HB 'no kit gt — skip auto gt'
    }
  }
  if (Test-Path $Gt) {
    & $PyHost $ScorePy --metric $MetricPy --gt $Gt `
      --a "v2docker=$JobDir\out_mock\result.geojson" `
      --out "$JobDir\scores_mock.json" *> "$JobDir\scores_mock.log"
    if (Test-Path $RefV2) {
      & $PyHost $ScorePy --metric $MetricPy --gt $Gt `
        --a "v2ref=$RefV2" --b "v2docker=$JobDir\out_mock\result.geojson" `
        --out "$JobDir\e2e_ref_vs_docker.json" *> "$JobDir\e2e_ref_vs_docker.log"
    }
  }
}

# honest-10: if train root + scoreboard available (optional; do not fail job if missing)
$Train = $env:KOZ3_TRAIN
if (-not $Train) { $Train = 'F:\2\train-expds-3' }
$Scoreboard = Get-ChildItem -Path $JobDir,$Z -Recurse -Filter 'scoreboard.py' -EA SilentlyContinue | Select-Object -First 1 -ExpandProperty FullName
if ((Test-Path $Train) -and $Scoreboard -and (Test-Path "$JobDir\out_mock\result.geojson")) {
  HB "honest-10 via $Scoreboard"
  New-Item -ItemType Directory -Force -Path "$JobDir\out_honest10" | Out-Null
  # Note: honest-10 normally needs per-site preds on train folds; if kit only has mock regions,
  # operator should re-run main on honest split or reuse kit honest pipeline. Record attempt.
  "honest-10 deferred: mock regions only; set KOZ3_HONEST_REGIONS and re-run or use kit honest flow" |
    Set-Content -Encoding UTF8 "$JobDir\out_honest10\README.txt"
  HB 'honest-10 placeholder written (need honest regions mount — see QUEUE)'
} else {
  HB 'honest-10 skipped (no train/scoreboard)'
  "skipped" | Set-Content "$JobDir\out_honest10_skipped.txt"
}

# Strict salvage: exit==0 AND geojson mtime after container start AND size>50 (LESSONS §1-2).
$geoPath = "$JobDir\out_mock\result.geojson"
$hasOut = Test-Path $geoPath
$gi = $null
if ($hasOut) { $gi = Get-Item $geoPath }
$fresh = $hasOut -and ($gi.Length -gt 50) -and ($gi.LastWriteTime -ge $t1)
$logOk = $false
if (Test-Path "$JobDir\main_mock.log") {
  if (Select-String -Path "$JobDir\main_mock.log" -Pattern '"written"' -Quiet -EA SilentlyContinue) { $logOk = $true }
  if (Select-String -Path "$JobDir\main_mock.log" -Pattern 'TOMLDecodeError|Invalid statement' -Quiet -EA SilentlyContinue) { $logOk = $false }
}
if (($ec -eq -1 -or $null -eq $ec) -and $fresh -and $logOk) {
  HB "resolve ExitCode $ec -> 0 (fresh geo+written log; Start-Process null quirk)"
  $ec = 0
}
$geoOk = $fresh -and ($ec -eq 0)
if ($hasOut) {
  HB ("geo size={0} mtime={1} start={2} ec={3} fresh={4} logOk={5} geoOk={6}" -f $gi.Length, $gi.LastWriteTime.ToString('o'), $t1.ToString('o'), $ec, $fresh, $logOk, $geoOk)
}
if (-not $geoOk) {
  if (-not $hasOut) { Set-Status "FAILED no_geojson docker_exit=$ec"; exit 5 }
  Set-Status "FAILED docker_exit=$ec geo_reject (need exit==0 and mtime>start and size>50)"; exit 4
}
Set-Status 'DONE'
HB 'all done'
Get-Content "$JobDir\warn_scan.txt" -EA SilentlyContinue | Select-Object -First 30
Get-Content "$JobDir\scores_mock.json" -EA SilentlyContinue | Select-Object -First 40
