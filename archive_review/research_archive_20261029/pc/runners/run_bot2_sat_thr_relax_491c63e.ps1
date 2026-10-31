# bot2 sat_thr_relax scale=0.85 ON vs OFF (v2 zip base + overlay from bots/bot2-sat-recall).
# Gate: pantum/n7q:20261024. Strict salvage: wipe stale geojson; DONE only if exit==0 and mtime>=start.
# Never touch talent/kontur/microphoenix containers. JobDir on C:\Sloi\pc_jobs only.
$ErrorActionPreference = 'Continue'
[Console]::OutputEncoding = [Text.Encoding]::UTF8
$JobDir = 'C:\Sloi\pc_jobs\bot2_sat_thr_relax_491c63e'
$Branch = 'bots/bot2-sat-recall'
$Sha = '491c63ec246a30949c21cd2c7675ba1ef75a0a34'
$Wt = 'C:\Sloi\wt_bot2_sat_thr_relax_491c63e'
$PyHost = 'C:\Sloi\expds-koz3_v0.2.0\expds-koz3\.venv\Scripts\python.exe'
$Image = 'pantum/n7q:20261024'
$Regions = 'F:\Sloi_pc_jobs_archive\v3e2e\regions'
$ZipV2 = 'F:\Sloi_pc_jobs_archive\v3e2e\kit\submission_bot7cand_v2_eb268d6.zip'
$ToolsTar = 'F:\Sloi_pc_jobs_archive\v3e2e\kit\tools_v3e2e.tar'
$RefV2 = 'F:\Sloi_pc_jobs_archive\v3e2e\out_v3fix2\res_v2\result.geojson'
if (-not (Test-Path $RefV2)) { $RefV2 = 'F:\Sloi_pc_jobs_archive\v3e2e\out\res_v2\result.geojson' }
$KitGt = 'F:\Sloi_pc_jobs_archive\v3e2e\out_v3fix2\gt.geojson'

New-Item -ItemType Directory -Force -Path $JobDir, "$JobDir\out_off", "$JobDir\out_on", "$JobDir\notes", "$JobDir\zip_off", "$JobDir\zip_on" | Out-Null

function HB($m){ Add-Content -Encoding utf8 "$JobDir\heartbeat.log" "$(Get-Date -Format o) $m" }
function Set-Status($s){ $l="$s $(Get-Date -Format o)"; Set-Content -Encoding utf8 "$JobDir\STATUS.txt" $l; HB $l }

$st = if (Test-Path "$JobDir\STATUS.txt") { Get-Content "$JobDir\STATUS.txt" -Raw } else { '' }
if ($st -match '^DONE') { Write-Host 'already DONE'; exit 0 }

Set-Status 'RUNNING'
HB "start sat_thr_relax sha=$Sha"

# worktree with sat_thr_relax code
if (-not (Test-Path "$Wt\.git")) {
  HB "worktree add $Wt @$Sha"
  & cmd /c "git -C C:\Sloi\wt_bots fetch origin $Branch" 2>&1 | Out-File "$JobDir\git_fetch.log"
  & cmd /c "git -C C:\Sloi\wt_bots worktree add --detach `"$Wt`" $Sha" 2>&1 | Out-File "$JobDir\git_wt.log"
} else {
  & cmd /c "git -C C:\Sloi\wt_bots fetch origin $Branch" 2>&1 | Out-File "$JobDir\git_fetch.log"
  & cmd /c "git -C `"$Wt`" checkout --detach $Sha" 2>&1 | Out-File "$JobDir\git_co.log"
}

if (-not (Test-Path $ZipV2)) { Set-Status "FAILED no_zip $ZipV2"; exit 2 }

# tools
if (-not (Test-Path "$JobDir\tools")) {
  New-Item -ItemType Directory -Force -Path "$JobDir\tools" | Out-Null
  tar -xf $ToolsTar -C $JobDir 2>$null
}
$ScorePy = Get-ChildItem -Path $JobDir -Recurse -Filter e2e_score.py -EA SilentlyContinue | Select-Object -First 1 -ExpandProperty FullName
$MetricPy = Get-ChildItem -Path $JobDir -Recurse -Filter compute_metrics.py -EA SilentlyContinue | Select-Object -First 1 -ExpandProperty FullName
HB "scorer=$ScorePy"

function Prepare-Zip([string]$tag, [bool]$relaxOn) {
  $Z = Join-Path $JobDir "zip_$tag"
  if (-not (Test-Path "$Z\inference\main.py")) {
    HB "expand zip -> $Z"
    Expand-Archive -Force -Path $ZipV2 -DestinationPath $Z
  }
  # overlay post.py / watchdog.py / default.toml from worktree
  foreach ($rel in @('expds\post.py','expds\watchdog.py','configs\default.toml')) {
    $src = Join-Path $Wt $rel
    if (Test-Path $src) {
      $dst = Join-Path $Z $rel
      New-Item -ItemType Directory -Force -Path (Split-Path $dst) | Out-Null
      Copy-Item -Force $src $dst
      HB "overlay $rel"
    }
  }
  $toml = Get-ChildItem -Path $Z -Recurse -Filter 'default.toml' -EA SilentlyContinue | Select-Object -First 1 -ExpandProperty FullName
  if (-not $toml) { throw "no default.toml in $Z" }
  $t = Get-Content -Raw -Encoding UTF8 $toml
  if ($relaxOn) {
    # force enable + scale (ASCII only)
    if ($t -notmatch '\[sat_thr_relax\]') {
      $t = $t.TrimEnd() + "`r`n`r`n[sat_thr_relax]`r`nenabled = true`r`nscale = 0.85`r`n"
    } else {
      $t = $t -replace 'enabled = false', 'enabled = true'
      if ($t -notmatch 'scale\s*=') {
        $t = $t -replace '(\[sat_thr_relax\][^\[]*)', "`$1scale = 0.85`r`n"
      } else {
        $t = $t -replace 'scale\s*=\s*[0-9.]+', 'scale = 0.85'
      }
    }
  } else {
    $t = $t -replace 'enabled = true', 'enabled = false'
  }
  Set-Content -Encoding UTF8 -Path $toml -Value $t -NoNewline
  Select-String -Path $toml -Pattern '\[sat_thr_relax\]|enabled =|scale =' |
    Select-Object -First 12 | ForEach-Object { $_.Line } |
    Set-Content -Encoding UTF8 "$JobDir\toml_${tag}_snip.txt"
  return $Z
}

function Run-Docker([string]$tag, [string]$Z, [string]$outRel) {
  $Name = "koz3-bot2-satrelax-$tag-" + (Get-Date -Format 'yyyyMMddHHmmss')
  $outDir = Join-Path $JobDir $outRel
  New-Item -ItemType Directory -Force -Path $outDir | Out-Null
  Remove-Item (Join-Path $outDir 'result.geojson') -Force -EA SilentlyContinue
  $ZMount = ($Z -replace '\\','/')
  $RegMount = ($Regions -replace '\\','/')
  $OutMount = ($JobDir -replace '\\','/')
  $log = Join-Path $JobDir "main_$tag.log"
  $err = Join-Path $JobDir "main_$tag.err"
  HB "docker run $Name out=$outRel"
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
    'python', 'inference/main.py', '/data/regions', "/data/output/$outRel/result.geojson"
  )
  $p = Start-Process -FilePath 'docker' -ArgumentList $dockerArgs -PassThru -NoNewWindow `
    -RedirectStandardOutput $log -RedirectStandardError $err
  while (-not $p.HasExited) {
    HB "docker $tag pid=$($p.Id) $([int]((Get-Date)-$t1).TotalSeconds)s"
    Start-Sleep 20
  }
  $null = $p.WaitForExit(5000)
  $p.Refresh()
  $sec = [math]::Round(((Get-Date)-$t1).TotalSeconds,1)
  $ec = $p.ExitCode
  if ($null -eq $ec) { HB "WARN ExitCode null $tag"; $ec = -1 }
  HB "docker $tag exit=$ec sec=$sec"
  Set-Content "$JobDir\wall_${tag}_sec.txt" $sec
  docker rm -f $Name 2>&1 | Out-Null

  $geoPath = Join-Path $outDir 'result.geojson'
  $hasOut = Test-Path $geoPath
  $gi = $null
  if ($hasOut) { $gi = Get-Item $geoPath }
  $fresh = $hasOut -and ($gi.Length -gt 50) -and ($gi.LastWriteTime -ge $t1)
  $logOk = $false
  if (Test-Path $log) {
    if (Select-String -Path $log -Pattern '"written"' -Quiet -EA SilentlyContinue) { $logOk = $true }
    if (Select-String -Path $log -Pattern 'TOMLDecodeError|Invalid statement' -Quiet -EA SilentlyContinue) { $logOk = $false }
  }
  if (($ec -eq -1 -or $null -eq $ec) -and $fresh -and $logOk) {
    HB "resolve ExitCode $ec -> 0 ($tag fresh geo+written)"
    $ec = 0
  }
  $geoOk = $fresh -and ($ec -eq 0)
  if ($hasOut) {
    HB ("$tag geo size={0} mtime={1} start={2} ec={3} fresh={4} logOk={5} geoOk={6}" -f $gi.Length, $gi.LastWriteTime.ToString('o'), $t1.ToString('o'), $ec, $fresh, $logOk, $geoOk)
  }
  if (-not $geoOk) {
    if (-not $hasOut) { Set-Status "FAILED no_geojson_$tag docker_exit=$ec"; exit 5 }
    Set-Status "FAILED $tag docker_exit=$ec geo_reject"; exit 4
  }
  return $geoPath
}

$Zoff = Prepare-Zip 'off' $false
$Zon  = Prepare-Zip 'on' $true
$geoOff = Run-Docker 'off' $Zoff 'out_off'
$geoOn  = Run-Docker 'on'  $Zon  'out_on'

# score
$Gt = "$JobDir\gt.geojson"
if (-not (Test-Path $Gt)) {
  if (Test-Path $KitGt) { Copy-Item -Force $KitGt $Gt; HB 'gt from kit' }
}
if ($ScorePy -and (Test-Path $Gt)) {
  & $PyHost $ScorePy --metric $MetricPy --gt $Gt `
    --a "off=$geoOff" --b "on=$geoOn" `
    --out "$JobDir\e2e_off_on.json" *> "$JobDir\e2e_off_on.log"
  if (Test-Path $RefV2) {
    & $PyHost $ScorePy --metric $MetricPy --gt $Gt `
      --a "v2ref=$RefV2" --b "on=$geoOn" `
      --out "$JobDir\e2e_v2_on.json" *> "$JobDir\e2e_v2_on.log"
    & $PyHost $ScorePy --metric $MetricPy --gt $Gt `
      --a "v2ref=$RefV2" --b "off=$geoOff" `
      --out "$JobDir\e2e_v2_off.json" *> "$JobDir\e2e_v2_off.log"
  }
}

Set-Status 'DONE'
HB 'all done'
Get-Content "$JobDir\e2e_off_on.json" -EA SilentlyContinue | Select-Object -First 60
Get-Content "$JobDir\toml_on_snip.txt" -EA SilentlyContinue
