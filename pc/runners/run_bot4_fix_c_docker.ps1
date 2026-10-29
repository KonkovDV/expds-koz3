# bot-4 Fix C: mock + honest-10 for dfine_ae_gor_area_tau alone.
# BOTH arms enable [dfine_aeor] ae (gor=1.01 baseline). ON adds Fix C sub-flag.
# Without Ae D-FINE, Fix C cannot fire -> Delta=0 (LESSONS). Docker ONLY pantum/n7q:20261024.
# JobDir/Wt/weights on C: (do not write F:). ASCII-only. No python -c. schtasks launch.
# Container name prefix koz3- only. Flags stay OFF in committed default.toml.
$ErrorActionPreference = 'Continue'
[Console]::OutputEncoding = [Text.Encoding]::UTF8

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$PreflightPy = Join-Path $ScriptDir 'bot3_ae_weights_preflight.py'
$SummaryPy   = Join-Path $ScriptDir 'bot4_lo_ablate_summary.py'

$Branch = 'bots/bot4-large-objects'
$Sha    = if ($env:KOZ3_SHA) { $env:KOZ3_SHA } else { 'e9d187a' }
$JobDir = if ($env:KOZ3_JOBDIR) { $env:KOZ3_JOBDIR } else { 'C:\Sloi\pc_jobs\bot4_fix_c_e9d187a' }
$Wt     = if ($env:KOZ3_WT) { $env:KOZ3_WT } else { 'C:\Sloi\pc_jobs\wt_bot4_fix_c_e9d187a' }
$WeightsAe = if ($env:KOZ3_WEIGHTS_AE) { $env:KOZ3_WEIGHTS_AE } else { 'C:\Sloi\pc_jobs\weights\dfine_ae_ft' }
$ShaPrefix = if ($env:KOZ3_AE_SHA_PREFIX) { $env:KOZ3_AE_SHA_PREFIX } else { '3e619878' }

$Image     = 'pantum/n7q:20261024'
$EnvTag    = 'ENV=docker:pantum/n7q:20261024'
$RegionsMock = 'F:\Sloi_pc_jobs_archive\v3e2e\regions'
$HonestRegions = if ($env:KOZ3_HONEST_REGIONS) { $env:KOZ3_HONEST_REGIONS } else { 'C:\Sloi\pc_jobs\bot3_honest10_regions' }
$ModelsFill = $null
foreach ($c in @(
  'C:\Sloi\wt_bot5_50f6f27\models',
  'C:\Sloi\pc_jobs\_models_fill',
  'C:\Sloi\pc_jobs\wt_bot5_gor_dense_e2e\models',
  'C:\Sloi\pc_jobs\wt_bot4_lo_ablate_5097c4a\models'
)) { if (Test-Path $c) { $ModelsFill = $c; break } }
$ZipV3 = 'C:\Sloi\pc_jobs\_vendor_zips\submission_bot7cand_v3on_0876090.zip'
$PyHost = 'C:\Sloi\expds-koz3_v0.2.0\expds-koz3\.venv\Scripts\python.exe'
$ToolsTar = 'F:\Sloi_pc_jobs_archive\v3e2e\kit\tools_v3e2e.tar'
$ToolsSrc = 'F:\Sloi_pc_jobs_archive\v3e2e\out\tools'
$KitGt = 'F:\Sloi_pc_jobs_archive\v3e2e\out\gt.geojson'
if (-not (Test-Path $KitGt)) { $KitGt = 'F:\Sloi_pc_jobs_archive\v3e2e\out_v3fix2\gt.geojson' }
$Repo = 'C:\Sloi\wt_bots'

New-Item -ItemType Directory -Force -Path $JobDir, "$JobDir\notes",
  "$JobDir\out_off", "$JobDir\out_on", "$JobDir\dump_on",
  "$JobDir\out_honest_off", "$JobDir\out_honest_on", "$JobDir\dump_honest_on" | Out-Null

function HB($m) {
  $line = "$(Get-Date -Format o) $m"
  Add-Content -Encoding ascii "$JobDir\heartbeat.log" $line
  Write-Host $line
}
function Set-Status($s) {
  $l = "$s $EnvTag $(Get-Date -Format o)"
  Set-Content -Encoding ascii "$JobDir\STATUS.txt" $l
  HB $l
}

$st = if (Test-Path "$JobDir\STATUS.txt") { Get-Content "$JobDir\STATUS.txt" -Raw } else { '' }
if ($st -match '^DONE') { Write-Host 'already DONE'; exit 0 }

Set-Status 'RUNNING'
HB "start FixC sha=$Sha weights=$WeightsAe job=$JobDir wt=$Wt"

if (-not (Test-Path $PreflightPy)) { Set-Status "FAILED no_preflight_py $PreflightPy"; exit 2 }
if (-not (Test-Path $SummaryPy))   { Set-Status "FAILED no_summary_py $SummaryPy"; exit 2 }

# --- tip worktree on C: ---
& cmd.exe /c "git -C `"$Repo`" fetch origin $Branch" *> "$JobDir\notes\git_fetch.log"
if (-not (Test-Path "$Wt\.git")) {
  & cmd.exe /c "git -C `"$Repo`" worktree add --detach `"$Wt`" $Sha" *> "$JobDir\notes\git_wt.log"
} else {
  & cmd.exe /c "git -C `"$Wt`" fetch origin $Branch" *> "$JobDir\notes\git_fetch2.log"
  & cmd.exe /c "git -C `"$Wt`" checkout --force --detach $Sha" *> "$JobDir\notes\git_wt.log"
}
$head = (& cmd.exe /c "git -C `"$Wt`" rev-parse HEAD").Trim()
HB "worktree HEAD=$head"
if ($head.Substring(0, [Math]::Min(7, $head.Length)) -ne $Sha.Substring(0, [Math]::Min(7, $Sha.Length)) -and $head -ne $Sha) {
  Set-Status "FAILED bad_sha want=$Sha got=$head"; exit 2
}

# --- models fill (pixgbm etc) ---
if ($ModelsFill -and (Test-Path $ModelsFill)) {
  HB "robocopy models fill from $ModelsFill"
  & robocopy $ModelsFill "$Wt\models" /E /XO /XN /XC /NFL /NDL /NJH /NJS /nc /ns /np *> "$JobDir\notes\robocopy_models.log"
}
if ((Test-Path $ZipV3) -and -not (Test-Path "$Wt\models\rerank_arch_lgbm.txt")) {
  HB "vendor zip models"
  & tar -xf $ZipV3 -C "$Wt" --strip-components=0 2> "$JobDir\notes\vendor_zip.err"
}

# --- Ae weights ---
New-Item -ItemType Directory -Force -Path "$Wt\models\dfine_ae", "$Wt\models\dfine_aeor" | Out-Null
$AeSrcSt = Join-Path $WeightsAe 'model.safetensors'
if (-not (Test-Path $AeSrcSt)) {
  # allow fold pack layout models/dfine_ae/model.safetensors under WeightsAe
  $alt = Join-Path $WeightsAe 'models\dfine_ae\model.safetensors'
  if (Test-Path $alt) { $WeightsAe = Join-Path $WeightsAe 'models\dfine_ae'; $AeSrcSt = Join-Path $WeightsAe 'model.safetensors' }
}
if (-not (Test-Path $AeSrcSt)) { Set-Status "FAILED missing_ae_weights $WeightsAe"; exit 3 }
if ((Get-Item $AeSrcSt).Length -le 0) { Set-Status "FAILED empty_ae_weights $AeSrcSt"; exit 3 }
Copy-Item -Force "$WeightsAe\*" "$Wt\models\dfine_ae\" -EA SilentlyContinue
# also copy nested model.safetensors if source is a dir of files
if (Test-Path $AeSrcSt) { Copy-Item -Force $AeSrcSt "$Wt\models\dfine_ae\model.safetensors" }
$ThrSrc = $null
foreach ($c in @(
  (Join-Path $WeightsAe 'thresholds_oof.json'),
  (Join-Path $WeightsAe 'models\dfine_aeor\thresholds_oof.json'),
  (Join-Path (Split-Path $WeightsAe -Parent) 'thresholds_oof.json'),
  (Join-Path (Split-Path $WeightsAe -Parent) 'models\dfine_aeor\thresholds_oof.json'),
  (Join-Path (Split-Path $WeightsAe -Parent) 'thresholds_ae_oof.json')
)) { if (Test-Path $c) { $ThrSrc = $c; break } }
if (-not $ThrSrc) { Set-Status "FAILED missing_thresholds near $WeightsAe"; exit 3 }
Copy-Item -Force $ThrSrc "$Wt\models\dfine_aeor\thresholds_oof.json"
HB "thresholds from $ThrSrc"

# Docker full-load preflight
Copy-Item -Force $PreflightPy "$JobDir\notes\bot3_ae_weights_preflight.py"
$wtM = ($Wt -replace '\\','/')
$jdM = ($JobDir -replace '\\','/')
$pfArgs = @(
  'run','--rm','--name',('koz3-bot4-fixc-preflight-' + (Get-Date -Format 'yyyyMMddHHmmss')),
  '--gpus','all','--network','none',
  '-e','PYTHONDONTWRITEBYTECODE=1','-e','PYTHONUNBUFFERED=1',
  '-e','HF_HUB_OFFLINE=1','-e','TRANSFORMERS_OFFLINE=1',
  '-v',"${wtM}/models/dfine_ae:/weights:ro",
  '-v',"${jdM}/notes/bot3_ae_weights_preflight.py:/preflight.py:ro",
  $Image,
  'python','/preflight.py','--dir','/weights','--sha-prefix',$ShaPrefix,'--require-config','--full-load'
)
HB "docker preflight Ae weights"
$pp = Start-Process -FilePath 'docker' -ArgumentList $pfArgs -PassThru -NoNewWindow `
  -RedirectStandardOutput "$JobDir\notes\preflight.out" -RedirectStandardError "$JobDir\notes\preflight.err"
Wait-Process -Id $pp.Id
if ($pp.ExitCode -ne 0) { Set-Status "FAILED ae_preflight exit=$($pp.ExitCode)"; exit 3 }

# empty weight stubs -> fail
$wbad = @()
Get-ChildItem -Path "$Wt\models" -Recurse -Include *.pt,*.pth,*.safetensors -EA SilentlyContinue | ForEach-Object {
  if ($_.Length -le 0) { $wbad += $_.FullName }
}
if ($wbad.Count -gt 0) {
  $wbad | Set-Content -Encoding ascii "$JobDir\notes\empty_weights.txt"
  Set-Status ("FAILED empty_weights " + ($wbad -join ';')); exit 3
}

# --- configs: BOTH arms enable dfine_aeor ae; ON adds Fix C ---
$Toml = "$Wt\configs\default.toml"
$BaseToml = [IO.File]::ReadAllText($Toml)
$Utf8 = [Text.UTF8Encoding]::new($false)
$AeBlock = @(
  '[dfine_aeor]',
  'enabled = true',
  'markups = ["ae"]',
  'weights = { ae = ["models/dfine_ae"] }',
  'thresholds_file = "models/dfine_aeor/thresholds_oof.json"',
  ''
) -join "`n"
if ($BaseToml -notmatch "\[dfine_aeor\]\r?\nenabled = false") {
  Set-Status 'FAILED toml_dfine_aeor_anchor_missing'; exit 6
}
if ($BaseToml -notmatch 'dfine_ae_gor_area_tau') {
  Set-Status 'FAILED toml_fix_c_flag_missing (need sha with Fix C)'; exit 6
}
$WithAe = [regex]::Replace($BaseToml, "\[dfine_aeor\]\r?\nenabled = false", $AeBlock)

function Patch-FixC([string]$text, [bool]$fixCOn) {
  # master + only dfine_ae_gor_area_tau; other LO subs stay false
  $en = if ($FixCOn) { 'true' } else { 'false' }
  $text = [regex]::Replace($text,
    '(?ms)(\[large_objects\](?:(?!^\[).)*?^)enabled\s*=\s*(?:true|false)',
    "`${1}enabled = $en", 1)
  foreach ($k in @('area_tau','protect_merge','shape_cores','fort_to_gor','coarse_hillfort','dfine_ae_gor_area_tau')) {
    $val = if ($FixCOn -and $k -eq 'dfine_ae_gor_area_tau') { 'true' } else { 'false' }
    $text = [regex]::Replace($text,
      "(?ms)(\[large_objects\](?:(?!^\[).)*?^)$k\s*=\s*(?:true|false)",
      "`${1}$k = $val", 1)
  }
  return $text
}
$OffToml = Patch-FixC $WithAe $false
$OnToml  = Patch-FixC $WithAe $true

function Run-Variant([string]$name, [string]$tomlText, [string]$regions, [string]$outSub, [switch]$Dump) {
  $res = "$JobDir\$outSub\result.geojson"
  if ((Test-Path $res) -and ((Get-Item $res).Length -gt 50)) { HB "skip $name (result exists)"; return 0 }
  New-Item -ItemType Directory -Force -Path "$JobDir\$outSub" | Out-Null
  [IO.File]::WriteAllText($Toml, $tomlText, $Utf8)
  Copy-Item -Force $Toml "$JobDir\$outSub\default.toml"
  Select-String -Path "$JobDir\$outSub\default.toml" -Pattern '\[dfine_aeor\]|\[large_objects\]|enabled =|dfine_ae_gor|markups|thresholds_file' |
    ForEach-Object { $_.Line } | Set-Content -Encoding ascii "$JobDir\$outSub\toml_snip.txt"
  $CName = "koz3-bot4-fixc-$name-" + (Get-Date -Format 'yyyyMMddHHmmss')
  $envArgs = @('-e','PYTHONDONTWRITEBYTECODE=1','-e','PYTHONUNBUFFERED=1',
    '-e','HF_HUB_OFFLINE=1','-e','TRANSFORMERS_OFFLINE=1','-e','EXPDS_STRICT=1',
    '-e',"EXPDS_STATUS_DIR=/data/output/$outSub")
  $dargs = @('run','--rm','--name',$CName,'--gpus','all','--network','none') + $envArgs + @(
    '-v',(($Wt -replace '\\','/') + ':/data/zip:ro'),
    '-v',(($regions -replace '\\','/') + ':/data/regions:ro'),
    '-v',(($JobDir -replace '\\','/') + ':/data/output'),
    '-w','/data/zip', $Image,
    'python','inference/main.py','/data/regions',"/data/output/$outSub/result.geojson"
  )
  HB "docker run $name $CName"
  $t1 = Get-Date
  $p = Start-Process -FilePath 'docker' -ArgumentList $dargs -PassThru -NoNewWindow `
    -RedirectStandardOutput "$JobDir\$outSub\main.log" -RedirectStandardError "$JobDir\$outSub\main.err"
  while (-not $p.HasExited) {
    HB "$name running $([int]((Get-Date) - $t1).TotalSeconds)s"
    Start-Sleep 20
  }
  $sec = [math]::Round(((Get-Date) - $t1).TotalSeconds, 1)
  $ec = $p.ExitCode; if ($null -eq $ec) { $ec = -1 }
  Set-Content -Encoding ascii "$JobDir\$outSub\wall_sec.txt" "$sec"
  HB "$name exit=$ec wall_sec=$sec"
  return $ec
}

# ========== (a) mock OFF + ON ==========
$null = Run-Variant 'off' $OffToml $RegionsMock 'out_off'
$null = Run-Variant 'on'  $OnToml  $RegionsMock 'out_on' -Dump
[IO.File]::WriteAllText($Toml, $BaseToml, $Utf8)

if (-not (Test-Path "$JobDir\out_off\result.geojson") -or -not (Test-Path "$JobDir\out_on\result.geojson")) {
  Set-Status 'FAILED missing_mock_geojson'; exit 4
}

# assert Ae dets > 0 on ON (and OFF — both have dfine_aeor)
$AssertPy = "$Wt\scripts\assert_dfine_aeor_ae_nonzero.py"
if (-not (Test-Path $AssertPy)) { Set-Status "FAILED no_assert $AssertPy"; exit 2 }
foreach ($arm in @('off','on')) {
  & $PyHost $AssertPy --geojson "$JobDir\out_$arm\result.geojson" --main-log "$JobDir\out_$arm\main.log" `
    *> "$JobDir\assert_ae_nonzero_$arm.log"
  if ($LASTEXITCODE -ne 0) { Set-Status "FAILED assert_ae_nonzero_$arm exit=$LASTEXITCODE"; exit 9 }
}

# scorers
New-Item -ItemType Directory -Force -Path "$JobDir\tools" | Out-Null
if (Test-Path $ToolsTar) { tar -xf $ToolsTar -C $JobDir 2>$null }
foreach ($f in @('e2e_score.py','compute_metrics.py')) {
  $dst = Join-Path $JobDir "tools\$f"
  if (-not (Test-Path $dst) -and (Test-Path $ToolsSrc)) {
    $src = Join-Path $ToolsSrc $f
    if (Test-Path $src) { Copy-Item -Force $src $dst }
  }
}
$ScorePy  = Get-ChildItem -Path $JobDir -Recurse -Filter e2e_score.py -EA SilentlyContinue | Select-Object -First 1 -ExpandProperty FullName
$MetricPy = Get-ChildItem -Path $JobDir -Recurse -Filter compute_metrics.py -EA SilentlyContinue | Select-Object -First 1 -ExpandProperty FullName
if (-not $ScorePy -or -not $MetricPy -or -not (Test-Path $KitGt)) { Set-Status 'FAILED no_scorer_or_gt'; exit 7 }

& $PyHost $ScorePy --metric $MetricPy --gt $KitGt `
  --a "off=$JobDir\out_off\result.geojson" --b "on=$JobDir\out_on\result.geojson" `
  --out "$JobDir\e2e_off_vs_on.json" *> "$JobDir\e2e_off_vs_on.log"

Copy-Item -Force $SummaryPy "$JobDir\notes\bot4_lo_ablate_summary.py"
& $PyHost $SummaryPy $JobDir 'off,on' *> "$JobDir\notes\summary.log"

# warn scan
Select-String -Path "$JobDir\out_on\main.log","$JobDir\out_off\main.log" `
  -Pattern 'dfine_aeor|large_objects|DFINE_AEOR_UNAVAILABLE|UNAVAILABLE|Traceback|ImportError' -EA SilentlyContinue |
  ForEach-Object { $_.Line } | Set-Content -Encoding ascii "$JobDir\warn_scan.txt"

# ========== (b) honest-10 if regions present ==========
if ((Test-Path $HonestRegions) -and (Get-ChildItem $HonestRegions -EA SilentlyContinue | Measure-Object).Count -gt 0) {
  HB "honest-10 regions=$HonestRegions"
  $null = Run-Variant 'honest_off' $OffToml $HonestRegions 'out_honest_off'
  $null = Run-Variant 'honest_on'  $OnToml  $HonestRegions 'out_honest_on'
  [IO.File]::WriteAllText($Toml, $BaseToml, $Utf8)
  if ((Test-Path "$JobDir\out_honest_on\result.geojson") -and (Test-Path "$JobDir\out_honest_off\result.geojson")) {
    & $PyHost $AssertPy --geojson "$JobDir\out_honest_on\result.geojson" --main-log "$JobDir\out_honest_on\main.log" `
      *> "$JobDir\assert_ae_nonzero_honest.log"
    if ($LASTEXITCODE -ne 0) { Set-Status "FAILED assert_ae_nonzero_honest exit=$LASTEXITCODE"; exit 9 }
    $HonestGt = Join-Path $HonestRegions '..\gt.geojson'
    if (-not (Test-Path $HonestGt)) { $HonestGt = $KitGt }
    & $PyHost $ScorePy --metric $MetricPy --gt $HonestGt `
      --a "honest_off=$JobDir\out_honest_off\result.geojson" `
      --b "honest_on=$JobDir\out_honest_on\result.geojson" `
      --out "$JobDir\scores_honest.json" *> "$JobDir\scores_honest.log"
  }
} else {
  HB "honest-10 SKIP (no regions at $HonestRegions)"
  Set-Content -Encoding ascii "$JobDir\notes\honest_skipped.txt" "no regions at $HonestRegions"
}

# restore default.toml
[IO.File]::WriteAllText($Toml, $BaseToml, $Utf8)

# Gate note: Delta==0 with Ae dets>0 may mean no large ae gor mid-conf on mock (not a silent miss).
Set-Status 'DONE'
HB 'DONE Fix C mock(+honest if any)'
exit 0
