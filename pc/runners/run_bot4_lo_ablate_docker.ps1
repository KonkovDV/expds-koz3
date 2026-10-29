# bot-4 large_objects mock ablation in Docker pantum/n7q (PROTOCOL section 10).
# Modeled on run_bot2_v2_base_docker.ps1 + run_bot5_gor_dense_e2e_docker.ps1.
# Host .venv ONLY for scoring. Gate inference MUST be docker.
# JobDir on C:\Sloi\pc_jobs (F: is low on space). Worktree on F:\Sloi_wt.
# Variants: off; on+each sub-flag alone; on+all_except_coarse. Idempotent per-run skip.
# Launch via schtasks (session-detached). ASCII-only. Summary via bot4_lo_ablate_summary.py.
# Container name prefix koz3- only.
$ErrorActionPreference = 'Continue'
[Console]::OutputEncoding = [Text.Encoding]::UTF8

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$SummaryPy = Join-Path $ScriptDir 'bot4_lo_ablate_summary.py'

$Sha       = '5097c4a'
$Branch    = 'bots/bot4-large-objects'
$JobDir    = 'C:\Sloi\pc_jobs\bot4_lo_ablate_5097c4a'
$Wt        = 'F:\Sloi_wt\wt_bot4_lo_ablate_5097c4a'
$Repo      = 'C:\Sloi\wt_bots'
$Image     = 'pantum/n7q:20261024'
$EnvTag    = 'ENV=docker:pantum/n7q:20261024'
$Regions   = 'F:\Sloi_pc_jobs_archive\v3e2e\regions'
$ToolsTar  = 'F:\Sloi_pc_jobs_archive\v3e2e\kit\tools_v3e2e.tar'
$ToolsSrc  = 'F:\Sloi_pc_jobs_archive\v3e2e\out\tools'
$KitGt     = 'F:\Sloi_pc_jobs_archive\v3e2e\out\gt.geojson'
if (-not (Test-Path $KitGt)) { $KitGt = 'F:\Sloi_pc_jobs_archive\v3e2e\out_v3fix2\gt.geojson' }
$PyHost    = 'C:\Sloi\expds-koz3_v0.2.0\expds-koz3\.venv\Scripts\python.exe'
$ModelsFillCandidates = @(
  'C:\Sloi\wt_bot4_lo_337a27c\models',
  'C:\Sloi\wt_bot5_50f6f27\models',
  'F:\Sloi_wt\wt_bot5_gor_dense_e2e\models'
)

$Variants = [ordered]@{
  'off'                  = @()
  'on_area_tau'          = @('area_tau')
  'on_protect_merge'     = @('protect_merge')
  'on_shape_cores'       = @('shape_cores')
  'on_fort_to_gor'       = @('fort_to_gor')
  'on_coarse_hillfort'   = @('coarse_hillfort')
  'on_all_except_coarse' = @('area_tau','protect_merge','shape_cores','fort_to_gor')
}

New-Item -ItemType Directory -Force -Path $JobDir, "$JobDir\notes" | Out-Null
foreach ($v in $Variants.Keys) {
  New-Item -ItemType Directory -Force -Path "$JobDir\out_$v" | Out-Null
}

function HB($m) {
  $line = "$(Get-Date -Format o) $m"
  Add-Content -Encoding utf8 "$JobDir\heartbeat.log" $line
  Write-Host $line
}
function Set-Status($s) {
  $l = "$s $EnvTag $(Get-Date -Format o)"
  Set-Content -Encoding utf8 "$JobDir\STATUS.txt" $l
  HB $l
}

$st = if (Test-Path "$JobDir\STATUS.txt") { Get-Content "$JobDir\STATUS.txt" -Raw } else { '' }
if ($st -match '^DONE') { Write-Host 'already DONE'; exit 0 }

Set-Status 'RUNNING'
HB "start sha=$Sha image=$Image job=$JobDir wt=$Wt"

if (-not (Test-Path $SummaryPy)) {
  Set-Status "FAILED no_summary_py $SummaryPy"
  exit 2
}

if (-not (Test-Path "$Repo\.git") -and -not (Test-Path "$Repo\HEAD")) {
  Set-Status "FAILED no_repo $Repo"; exit 2
}
if (-not (Test-Path "$Wt\.git")) {
  & cmd.exe /c "git -C `"$Repo`" fetch origin $Branch" *> "$JobDir\notes\git_fetch.log"
  & cmd.exe /c "git -C `"$Repo`" worktree add --detach `"$Wt`" $Sha" *> "$JobDir\notes\git_wt.log"
} else {
  & cmd.exe /c "git -C `"$Repo`" fetch origin $Branch" *> "$JobDir\notes\git_fetch.log"
  & cmd.exe /c "git -C `"$Wt`" fetch origin $Branch" *> "$JobDir\notes\git_fetch2.log"
  & cmd.exe /c "git -C `"$Wt`" checkout --force --detach $Sha" *> "$JobDir\notes\git_co.log"
}
$head = (& cmd.exe /c "git -C `"$Wt`" rev-parse HEAD").Trim()
HB "worktree HEAD=$head want=$Sha"
if (-not ($head.StartsWith($Sha))) {
  Set-Status "FAILED bad_sha want=$Sha got=$head"
  exit 2
}

$ModelsFill = $null
foreach ($c in $ModelsFillCandidates) {
  if (Test-Path $c) { $ModelsFill = $c; break }
}
if ($ModelsFill) {
  HB "robocopy models from $ModelsFill"
  & robocopy $ModelsFill "$Wt\models" /E /XO /NFL /NDL /NJH /NJS *> "$JobDir\notes\robocopy_models.log"
} else {
  HB 'WARN no ModelsFill - relying on worktree models'
}
if (-not (Test-Path "$Wt\inference\main.py")) { Set-Status 'FAILED no_inference_main'; exit 3 }
if (-not (Test-Path "$Wt\configs\default.toml")) { Set-Status 'FAILED no_default_toml'; exit 3 }

$TomlPath = "$Wt\configs\default.toml"
$Utf8 = [Text.UTF8Encoding]::new($false)
$BaseToml = [IO.File]::ReadAllText($TomlPath)
if ($BaseToml -notmatch '\[large_objects\]') { Set-Status 'FAILED no_large_objects_section'; exit 6 }
[IO.File]::WriteAllText($TomlPath, $BaseToml, $Utf8)

function Patch-LoToml([string]$text, [bool]$masterOn, [string[]]$subsOn) {
  $all = @('area_tau','protect_merge','shape_cores','fort_to_gor','coarse_hillfort')
  $en = if ($masterOn) { 'true' } else { 'false' }
  $text = [regex]::Replace($text,
    '(?ms)(\[large_objects\](?:(?!^\[).)*?^)enabled\s*=\s*(?:true|false)',
    "`${1}enabled = $en", 1)
  foreach ($k in $all) {
    $val = if ($subsOn -contains $k) { 'true' } else { 'false' }
    $text = [regex]::Replace($text,
      "(?ms)(\[large_objects\](?:(?!^\[).)*?^)$k\s*=\s*(?:true|false)",
      "`${1}$k = $val", 1)
  }
  return $text
}

function Run-Variant([string]$name, [bool]$masterOn, [string[]]$subsOn) {
  $outDir = "$JobDir\out_$name"
  $res = "$outDir\result.geojson"
  if ((Test-Path $res) -and ((Get-Item $res).Length -gt 50)) {
    HB "skip $name (result exists)"
    return 0
  }
  New-Item -ItemType Directory -Force -Path $outDir | Out-Null
  $tomlText = Patch-LoToml $BaseToml $masterOn $subsOn
  [IO.File]::WriteAllText($TomlPath, $tomlText, $Utf8)
  Copy-Item -Force $TomlPath "$outDir\default.toml"
  Select-String -Path "$outDir\default.toml" -Pattern '\[large_objects\]|enabled =|area_tau|protect_merge|shape_cores|fort_to_gor|coarse_hillfort' |
    ForEach-Object { $_.Line } | Set-Content -Encoding utf8 "$outDir\toml_snip.txt"

  $CName = "koz3-bot4-lo-$name-" + (Get-Date -Format 'yyyyMMddHHmmss')
  $WtM = ($Wt -replace '\\','/')
  $RegM = ($Regions -replace '\\','/')
  $OutM = ($JobDir -replace '\\','/')
  $dargs = @(
    'run','--rm','--name', $CName,
    '--gpus','all',
    '--network','none',
    '-e','PYTHONDONTWRITEBYTECODE=1',
    '-e','PYTHONUNBUFFERED=1',
    '-v', "${WtM}:/data/zip:ro",
    '-v', "${RegM}:/data/regions:ro",
    '-v', "${OutM}:/data/output",
    '-w', '/data/zip',
    $Image,
    'python', 'inference/main.py', '/data/regions', "/data/output/out_$name/result.geojson"
  )
  HB "docker run $name $CName"
  $t1 = Get-Date
  $p = Start-Process -FilePath 'docker' -ArgumentList $dargs -PassThru -NoNewWindow `
    -RedirectStandardOutput "$outDir\main.log" -RedirectStandardError "$outDir\main.err"
  while (-not $p.HasExited) {
    HB "$name running $([int]((Get-Date) - $t1).TotalSeconds)s"
    Start-Sleep 20
  }
  $sec = [math]::Round(((Get-Date) - $t1).TotalSeconds, 1)
  $ec = $p.ExitCode; if ($null -eq $ec) { $ec = -1 }
  Set-Content -Encoding utf8 "$outDir\wall_sec.txt" "$sec"
  HB "$name exit=$ec wall_sec=$sec"
  return $ec
}

$fail = @()
foreach ($name in $Variants.Keys) {
  $subs = @($Variants[$name])
  $master = ($name -ne 'off')
  $ec = Run-Variant $name $master $subs
  if (-not (Test-Path "$JobDir\out_$name\result.geojson")) {
    $fail += $name
    HB "MISSING geojson $name exit=$ec"
  } elseif (($ec -ne 0) -and ($ec -ne -1)) {
    HB "WARN $name exit=$ec but geojson present - continue"
  }
}
[IO.File]::WriteAllText($TomlPath, $BaseToml, $Utf8)

if ($fail.Count -gt 0) {
  Set-Status ("FAILED missing_geojson " + ($fail -join ','))
  exit 4
}

New-Item -ItemType Directory -Force -Path "$JobDir\tools" | Out-Null
if (Test-Path $ToolsTar) { tar -xf $ToolsTar -C $JobDir 2>$null }
foreach ($f in @('e2e_score.py','compute_metrics.py')) {
  $dst = Join-Path $JobDir "tools\$f"
  if (-not (Test-Path $dst) -and (Test-Path $ToolsSrc)) {
    $src = Join-Path $ToolsSrc $f
    if (Test-Path $src) { Copy-Item -Force $src $dst; HB "copied scorer $f" }
  }
}
$ScorePy  = Get-ChildItem -Path $JobDir -Recurse -Filter e2e_score.py -EA SilentlyContinue | Select-Object -First 1 -ExpandProperty FullName
$MetricPy = Get-ChildItem -Path $JobDir -Recurse -Filter compute_metrics.py -EA SilentlyContinue | Select-Object -First 1 -ExpandProperty FullName
HB "scorer=$ScorePy metric=$MetricPy gt=$KitGt"
if (-not $ScorePy -or -not $MetricPy -or -not (Test-Path $KitGt)) {
  Set-Status 'FAILED no_scorer_or_gt'; exit 7
}
if (-not (Test-Path $PyHost)) { Set-Status "FAILED no_PyHost $PyHost"; exit 7 }

$offGeo = "$JobDir\out_off\result.geojson"
foreach ($name in $Variants.Keys) {
  if ($name -eq 'off') { continue }
  $onGeo = "$JobDir\out_$name\result.geojson"
  $outJ = "$JobDir\e2e_off_vs_$name.json"
  if (Test-Path $outJ) { HB "skip score $name (json exists)"; continue }
  HB "score off vs $name"
  & $PyHost $ScorePy --metric $MetricPy --gt $KitGt `
    --a "off=$offGeo" --b "${name}=$onGeo" `
    --out $outJ *> "$JobDir\e2e_off_vs_$name.log"
}

$varList = ($Variants.Keys -join ',')
Copy-Item -Force $SummaryPy "$JobDir\notes\bot4_lo_ablate_summary.py"
& $PyHost $SummaryPy $JobDir $varList *> "$JobDir\notes\summary.log"
if (-not (Test-Path "$JobDir\summary_layers.csv")) {
  Set-Status 'FAILED summary'
  Get-Content "$JobDir\notes\summary.log" -EA SilentlyContinue
  exit 8
}

Select-String -Path "$JobDir\out_*\main.log","$JobDir\out_*\main.err" `
  -Pattern 'large_objects|coarse hillfort|fort|UNAVAILABLE|Traceback|platform' -EA SilentlyContinue |
  ForEach-Object { "$($_.Filename): $($_.Line)" } |
  Set-Content -Encoding utf8 "$JobDir\warn_scan.txt"

Set-Status 'DONE'
HB 'all done'
Get-Content "$JobDir\SUMMARY.txt" -EA SilentlyContinue
