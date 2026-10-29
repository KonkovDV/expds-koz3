# bot3 Ae-ft: mock e2e + honest-10 for [dfine_aeor] Ae-only ON vs OFF.
# Docker ONLY: pantum/n7q:20261024 (PROTOCOL section 10). Host .venv = score/assert only.
# JobDir/Wt/weights on C: (F: ~3.5 GB free - do not write there). Inputs READ from F: ok.
# ASCII-only (PS5 ANSI-safe). No python -c. Launch via schtasks wrappers.
# Variants: -Variant final (sha 3e619878) | -Variant f4 (sha 249b86fb).
# After ON: scripts/assert_dfine_aeor_ae_nonzero.py HARD FAIL if n_raw/Ae dets == 0.
# Container name prefix koz3- only. [dfine_aeor] stays default OFF in configs/default.toml.
param(
  [ValidateSet('final','f4')]
  [string]$Variant = 'final',
  [string]$WeightsAe = '',
  [string]$ShaPrefix = '',
  [string]$JobDir = '',
  [string]$Wt = ''
)

$ErrorActionPreference = 'Continue'
[Console]::OutputEncoding = [Text.Encoding]::UTF8

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$PreflightPy = Join-Path $ScriptDir 'bot3_ae_weights_preflight.py'

$Branch = 'bots/bot3-dfine-aeor-eval'
$Sha    = 'PLACEHOLDER_SHA'
if ($env:KOZ3_SHA) { $Sha = $env:KOZ3_SHA }

if ($Variant -eq 'f4') {
  if (-not $JobDir)    { $JobDir = 'C:\Sloi\pc_jobs\bot3_dfine_aeor_ae_ft_f4' }
  if (-not $Wt)        { $Wt = 'C:\Sloi\pc_jobs\wt_bot3_dfine_aeor_ae_ft_f4' }
  if (-not $WeightsAe) { $WeightsAe = 'C:\Sloi\pc_jobs\weights\dfine_ae_ft_f4' }
  if (-not $ShaPrefix) { $ShaPrefix = '249b86fb' }
} else {
  if (-not $JobDir)    { $JobDir = 'C:\Sloi\pc_jobs\bot3_dfine_aeor_ae_ft' }
  if (-not $Wt)        { $Wt = 'C:\Sloi\pc_jobs\wt_bot3_dfine_aeor_ae_ft' }
  if (-not $WeightsAe) { $WeightsAe = 'C:\Sloi\pc_jobs\weights\dfine_ae_ft' }
  if (-not $ShaPrefix) { $ShaPrefix = '3e619878' }
}
if ($env:KOZ3_WEIGHTS_AE) { $WeightsAe = $env:KOZ3_WEIGHTS_AE }
if ($env:KOZ3_AE_SHA_PREFIX) { $ShaPrefix = $env:KOZ3_AE_SHA_PREFIX }

$Image     = 'pantum/n7q:20261024'
$EnvTag    = 'ENV=docker:pantum/n7q:20261024'
$RegionsMock = 'F:\Sloi_pc_jobs_archive\v3e2e\regions'
$HonestRegions = if ($env:KOZ3_HONEST_REGIONS) { $env:KOZ3_HONEST_REGIONS } else { 'C:\Sloi\pc_jobs\bot3_honest10_regions' }
$ModelsFill = $null
foreach ($c in @(
  'C:\Sloi\wt_bot5_50f6f27\models',
  'C:\Sloi\pc_jobs\_models_fill',
  'C:\Sloi\pc_jobs\wt_bot5_gor_dense_e2e\models'
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
HB "start variant=$Variant sha=$Sha prefix=$ShaPrefix weights=$WeightsAe job=$JobDir wt=$Wt"

if (-not (Test-Path $PreflightPy)) {
  Set-Status "FAILED no_preflight_py $PreflightPy"
  exit 2
}

# --- resolve tip, then worktree on C: ---
& cmd.exe /c "git -C `"$Repo`" fetch origin $Branch" *> "$JobDir\notes\git_fetch.log"
if ($Sha -eq 'PLACEHOLDER_SHA') {
  $Sha = (& cmd.exe /c "git -C `"$Repo`" rev-parse origin/$Branch").Trim()
  if (-not $Sha) { Set-Status "FAILED resolve_sha origin/$Branch"; exit 2 }
  HB "resolved Sha from origin/$Branch -> $Sha"
}
if (-not (Test-Path "$Wt\.git")) {
  & cmd.exe /c "git -C `"$Repo`" worktree add --detach `"$Wt`" $Sha" *> "$JobDir\notes\git_wt.log"
} else {
  & cmd.exe /c "git -C `"$Wt`" fetch origin $Branch" *> "$JobDir\notes\git_fetch2.log"
  & cmd.exe /c "git -C `"$Wt`" checkout --force --detach $Sha" *> "$JobDir\notes\git_wt.log"
}
$head = (& cmd.exe /c "git -C `"$Wt`" rev-parse HEAD").Trim()
HB "worktree HEAD=$head"
if ($head -ne $Sha) { Set-Status "FAILED bad_sha want=$Sha got=$head"; exit 2 }

# --- models fill ---
if ($ModelsFill) {
  Get-ChildItem "$Wt\models" -Recurse -File -EA SilentlyContinue |
    Where-Object { $_.Length -eq 0 -and $_.Extension -match '\.(pt|pth|safetensors|onnx|bin)$' } |
    ForEach-Object { HB "remove_empty $($_.FullName)"; Remove-Item -Force $_.FullName }
  & robocopy $ModelsFill "$Wt\models" /E /IS /IT /NFL /NDL /NJH /NJS *> "$JobDir\notes\robocopy_models.log"
  HB "models filled from $ModelsFill"
}

# --- Ae weights into worktree models/dfine_ae ---
New-Item -ItemType Directory -Force -Path "$Wt\models\dfine_ae", "$Wt\models\dfine_aeor" | Out-Null
$AeSrcSt = Join-Path $WeightsAe 'model.safetensors'
if (-not (Test-Path $AeSrcSt)) {
  $alt = Join-Path $WeightsAe 'dfine_ae\model.safetensors'
  if (Test-Path $alt) { $WeightsAe = Join-Path $WeightsAe 'dfine_ae'; $AeSrcSt = Join-Path $WeightsAe 'model.safetensors' }
}
if (-not (Test-Path $AeSrcSt)) {
  Set-Status "FAILED missing_ae_weights $WeightsAe (expect model.safetensors)"
  exit 3
}
if ((Get-Item $AeSrcSt).Length -le 0) {
  Set-Status "FAILED empty_ae_weights $AeSrcSt"
  exit 3
}
& robocopy $WeightsAe "$Wt\models\dfine_ae" config.json model.safetensors /NFL /NDL /NJH /NJS *> "$JobDir\notes\robocopy_dfine_ae.log"
# thresholds
$ThrCandidates = @(
  (Join-Path $WeightsAe 'thresholds_oof.json'),
  (Join-Path $WeightsAe 'models\dfine_aeor\thresholds_oof.json'),
  (Join-Path (Split-Path $WeightsAe -Parent) 'thresholds_oof.json'),
  (Join-Path (Split-Path $WeightsAe -Parent) 'models\dfine_aeor\thresholds_oof.json')
)
$ThrSrc = $null
foreach ($t in $ThrCandidates) { if (Test-Path $t) { $ThrSrc = $t; break } }
if (-not $ThrSrc) { Set-Status "FAILED missing_thresholds near $WeightsAe"; exit 3 }
Copy-Item -Force $ThrSrc "$Wt\models\dfine_aeor\thresholds_oof.json"
HB "thresholds from $ThrSrc"

# --- HOST preflight: size + sha prefix + safetensors load (helper .py, no python -c) ---
if (-not (Test-Path $PyHost)) { Set-Status "FAILED no_PyHost $PyHost"; exit 3 }
Copy-Item -Force $PreflightPy "$JobDir\notes\bot3_ae_weights_preflight.py"
$hostPf = & $PyHost $PreflightPy --dir "$Wt\models\dfine_ae" --sha-prefix $ShaPrefix --require-config 2>&1
$hostPf | Set-Content -Encoding ascii "$JobDir\notes\host_preflight.log"
HB "host_preflight $hostPf"
if ($LASTEXITCODE -ne 0) { Set-Status "FAILED host_preflight exit=$LASTEXITCODE"; exit 3 }

# --- DOCKER preflight: same helper inside pantum/n7q (hard FAIL) ---
$PfOut = "$JobDir\notes\docker_preflight"
New-Item -ItemType Directory -Force -Path $PfOut | Out-Null
$CNamePf = 'koz3-bot3-ae-preflight-' + (Get-Date -Format 'yyyyMMddHHmmss')
$wtM = ($Wt -replace '\\','/')
$jdM = ($JobDir -replace '\\','/')
$pfArgs = @(
  'run','--rm','--name',$CNamePf,'--network','none',
  '-e','PYTHONDONTWRITEBYTECODE=1','-e','PYTHONUNBUFFERED=1',
  '-v',"${wtM}/models/dfine_ae:/weights:ro",
  '-v',"${jdM}/notes/bot3_ae_weights_preflight.py:/preflight.py:ro",
  '-v',"${jdM}/notes/docker_preflight:/out",
  $Image,
  'python','/preflight.py','--dir','/weights','--sha-prefix',$ShaPrefix,'--require-config'
)
HB "docker preflight $CNamePf"
$tpf = Get-Date
$pp = Start-Process -FilePath 'docker' -ArgumentList $pfArgs -PassThru -NoNewWindow `
  -RedirectStandardOutput "$PfOut\stdout.log" -RedirectStandardError "$PfOut\stderr.log"
while (-not $pp.HasExited) { Start-Sleep 5 }
HB "docker_preflight exit=$($pp.ExitCode) sec=$([math]::Round(((Get-Date)-$tpf).TotalSeconds,1))"
Get-Content "$PfOut\stdout.log","$PfOut\stderr.log" -EA SilentlyContinue | Set-Content -Encoding ascii "$JobDir\notes\docker_preflight.log"
if ($pp.ExitCode -ne 0) {
  Set-Status "FAILED docker_preflight exit=$($pp.ExitCode)"
  exit 3
}
HB 'docker_preflight PASS'

# --- vendor manylinux ---
if (-not (Test-Path "$Wt\inference\lib\vendor\transformers")) {
  if (-not (Test-Path $ZipV3)) { Set-Status "FAILED no_vendor_zip $ZipV3"; exit 3 }
  $Tmp = Join-Path $JobDir 'vendor_unpack'
  New-Item -ItemType Directory -Force -Path $Tmp | Out-Null
  & tar -xf $ZipV3 -C $Tmp --wildcards 'inference/lib/vendor/*' *> "$JobDir\notes\vendor_tar.log"
  if (-not (Test-Path "$Tmp\inference\lib\vendor\transformers")) {
    Expand-Archive -Path $ZipV3 -DestinationPath $Tmp -Force
  }
  if (-not (Test-Path "$Tmp\inference\lib\vendor\transformers")) {
    Set-Status 'FAILED vendor_not_in_zip'; exit 3
  }
  Copy-Item -Recurse -Force "$Tmp\inference\lib\vendor" "$Wt\inference\lib\vendor"
  HB 'vendor copied'
}

# empty weight guard on all models
$wbad = @()
Get-ChildItem "$Wt\models" -Recurse -File -EA SilentlyContinue |
  Where-Object { $_.Extension -match '\.(pt|pth|safetensors)$' -and $_.Length -le 0 } |
  ForEach-Object { $wbad += $_.FullName }
if ($wbad.Count -gt 0) {
  $wbad | Set-Content -Encoding ascii "$JobDir\notes\empty_weights.txt"
  Set-Status ("FAILED empty_weights " + ($wbad -join ';'))
  exit 3
}

# --- configs: OFF = default.toml; ON = Ae-only dfine_aeor ---
$Toml = "$Wt\configs\default.toml"
$OffToml = [IO.File]::ReadAllText($Toml)
$OnBlock = @(
  '[dfine_aeor]',
  'enabled = true',
  'markups = ["ae"]',
  'weights = { ae = ["models/dfine_ae"] }',
  'thresholds_file = "models/dfine_aeor/thresholds_oof.json"',
  ''
) -join "`n"
if ($OffToml -notmatch "\[dfine_aeor\]\r?\nenabled = false") {
  Set-Status 'FAILED toml_anchor_missing'
  exit 6
}
$OnToml = [regex]::Replace($OffToml, "\[dfine_aeor\]\r?\nenabled = false", $OnBlock)
$Utf8 = [Text.UTF8Encoding]::new($false)

function Run-Variant([string]$name, [string]$tomlText, [string]$regions, [string]$outSub, [switch]$Dump) {
  $res = "$JobDir\$outSub\result.geojson"
  if (Test-Path $res) { HB "skip $name (result exists)"; return 0 }
  New-Item -ItemType Directory -Force -Path "$JobDir\$outSub" | Out-Null
  [IO.File]::WriteAllText($Toml, $tomlText, $Utf8)
  Copy-Item -Force $Toml "$JobDir\$outSub\default.toml"
  $CName = "koz3-bot3-aeor-$Variant-$name-" + (Get-Date -Format 'yyyyMMddHHmmss')
  $envArgs = @('-e','PYTHONDONTWRITEBYTECODE=1','-e','PYTHONUNBUFFERED=1',
    '-e','HF_HUB_OFFLINE=1','-e','TRANSFORMERS_OFFLINE=1','-e','EXPDS_STRICT=1',
    '-e',"EXPDS_STATUS_DIR=/data/output/$outSub")
  if ($Dump) {
    $dumpSub = ($outSub -replace '^out_','dump_')
    New-Item -ItemType Directory -Force -Path "$JobDir\$dumpSub" | Out-Null
    $envArgs += @('-e',"EXPDS_DFINE_DUMP=/data/output/$dumpSub")
  }
  $args2 = @('run','--rm','--name',$CName,'--gpus','all','--network','none') + $envArgs + @(
    '-v',"$(($Wt -replace '\\','/')):/data/zip:ro",
    '-v',"$(($regions -replace '\\','/')):/data/regions:ro",
    '-v',"$(($JobDir -replace '\\','/')):/data/output",
    '-w','/data/zip',$Image,
    'python','inference/main.py','/data/regions',"/data/output/$outSub/result.geojson")
  HB "docker run $name $CName"
  $t1 = Get-Date
  $p = Start-Process -FilePath 'docker' -ArgumentList $args2 -PassThru -NoNewWindow `
    -RedirectStandardOutput "$JobDir\$outSub\main.log" -RedirectStandardError "$JobDir\$outSub\main.err"
  while (-not $p.HasExited) {
    HB "$name running $([int]((Get-Date)-$t1).TotalSeconds)s"
    Start-Sleep 20
  }
  $sec = [math]::Round(((Get-Date)-$t1).TotalSeconds,1)
  Set-Content -Encoding ascii "$JobDir\$outSub\wall_sec.txt" "$sec"
  HB "$name exit=$($p.ExitCode) wall_sec=$sec"
  return $p.ExitCode
}

# ========== (a) mock e2e OFF + ON ==========
if (-not (Test-Path $RegionsMock)) { Set-Status "FAILED no_mock_regions $RegionsMock"; exit 3 }
$ecOff = Run-Variant 'mock_off' $OffToml $RegionsMock 'out_off'
$ecOn  = Run-Variant 'mock_on'  $OnToml  $RegionsMock 'out_on' -Dump
[IO.File]::WriteAllText($Toml, $OffToml, $Utf8)

Select-String -Path "$JobDir\out_off\main.log","$JobDir\out_on\main.log","$JobDir\out_off\main.err","$JobDir\out_on\main.err" `
  -Pattern 'dfine_aeor|DFINE_AEOR|SP_DFINE_UNAVAILABLE|UNAVAILABLE|ImportError|Traceback|n_det=' -EA SilentlyContinue |
  ForEach-Object { "$($_.Filename): $($_.Line)" } |
  Set-Content -Encoding ascii "$JobDir\warn_scan.txt"

if (-not (Test-Path "$JobDir\out_off\result.geojson")) { Set-Status "FAILED off exit=$ecOff no_geojson"; exit 4 }
if (-not (Test-Path "$JobDir\out_on\result.geojson"))  { Set-Status "FAILED on exit=$ecOn no_geojson"; exit 5 }

# HARD FAIL: Ae dets / n_raw > 0
$AssertPy = "$Wt\scripts\assert_dfine_aeor_ae_nonzero.py"
if (-not (Test-Path $AssertPy)) { Set-Status "FAILED no_assert_py $AssertPy"; exit 9 }
$AssertArgs = @($AssertPy,
  '--geojson', "$JobDir\out_on\result.geojson",
  '--main-log', "$JobDir\out_on\main.log")
$DumpJson = Get-ChildItem -Path "$JobDir\dump_on" -Recurse -Filter 'stages.json' -EA SilentlyContinue |
  Select-Object -First 1 -ExpandProperty FullName
if ($DumpJson) { $AssertArgs += @('--dump-json', $DumpJson) }
HB "assert: $($AssertArgs -join ' ')"
& $PyHost @AssertArgs *> "$JobDir\assert_ae_nonzero.log"
if ($LASTEXITCODE -ne 0) { Set-Status "FAILED assert_ae_nonzero exit=$LASTEXITCODE"; exit 9 }
HB 'assert Ae nonzero PASS'

# score mock OFF vs ON
New-Item -ItemType Directory -Force -Path "$JobDir\tools" | Out-Null
if (Test-Path $ToolsTar) { tar -xf $ToolsTar -C $JobDir 2>$null }
foreach ($f in @('e2e_score.py','compute_metrics.py')) {
  $dst = Join-Path $JobDir "tools\$f"
  if (-not (Test-Path $dst) -and (Test-Path (Join-Path $ToolsSrc $f))) {
    Copy-Item -Force (Join-Path $ToolsSrc $f) $dst
  }
}
$ScorePy  = Get-ChildItem -Path $JobDir -Recurse -Filter e2e_score.py -EA SilentlyContinue | Select-Object -First 1 -ExpandProperty FullName
$MetricPy = Get-ChildItem -Path $JobDir -Recurse -Filter compute_metrics.py -EA SilentlyContinue | Select-Object -First 1 -ExpandProperty FullName
if (-not $ScorePy -or -not $MetricPy -or -not (Test-Path $KitGt)) {
  Set-Status 'FAILED no_scorer_or_gt'
  exit 7
}
& $PyHost $ScorePy --metric $MetricPy --gt $KitGt `
  --a "ae_off_docker=$JobDir\out_off\result.geojson" `
  --b "dfine_aeor_ae_on=$JobDir\out_on\result.geojson" `
  --out "$JobDir\scores_mock.json" *> "$JobDir\scores_mock.log"
if (-not (Test-Path "$JobDir\scores_mock.json")) { Set-Status 'FAILED score_mock'; exit 8 }
HB 'mock score OK'

# ========== (b) honest-10 OFF + ON ==========
$honestRan = $false
if ((Test-Path $HonestRegions) -and (Get-ChildItem $HonestRegions -EA SilentlyContinue | Measure-Object).Count -gt 0) {
  HB "honest-10 regions=$HonestRegions"
  $null = Run-Variant 'honest_off' $OffToml $HonestRegions 'out_honest_off'
  $null = Run-Variant 'honest_on'  $OnToml  $HonestRegions 'out_honest_on' -Dump
  [IO.File]::WriteAllText($Toml, $OffToml, $Utf8)
  if ((Test-Path "$JobDir\out_honest_on\result.geojson") -and (Test-Path "$JobDir\out_honest_off\result.geojson")) {
    $AssertH = @($AssertPy,
      '--geojson', "$JobDir\out_honest_on\result.geojson",
      '--main-log', "$JobDir\out_honest_on\main.log")
    & $PyHost @AssertH *> "$JobDir\assert_ae_nonzero_honest.log"
    if ($LASTEXITCODE -ne 0) { Set-Status "FAILED assert_ae_nonzero_honest exit=$LASTEXITCODE"; exit 9 }
    $HonestGt = Join-Path $HonestRegions '..\gt.geojson'
    if (-not (Test-Path $HonestGt)) { $HonestGt = $KitGt }
    & $PyHost $ScorePy --metric $MetricPy --gt $HonestGt `
      --a "honest_off=$JobDir\out_honest_off\result.geojson" `
      --b "honest_ae_on=$JobDir\out_honest_on\result.geojson" `
      --out "$JobDir\scores_honest.json" *> "$JobDir\scores_honest.log"
    $honestRan = $true
  }
} else {
  @(
    "honest-10 deferred: no regions at $HonestRegions",
    "Prepare ~10 ae family sites then set KOZ3_HONEST_REGIONS and re-run.",
    "Mock phase already asserts n_raw>0 and scores OFF vs ON."
  ) | Set-Content -Encoding ascii "$JobDir\out_honest_off\README.txt"
  HB 'honest-10 deferred (no regions)'
}

Set-Status 'DONE'
Get-Content "$JobDir\assert_ae_nonzero.log" -EA SilentlyContinue | Select-Object -First 40
if (Test-Path "$JobDir\scores_mock.json") { Get-Content "$JobDir\scores_mock.json" | Select-Object -First 80 }
HB "honest_ran=$honestRan variant=$Variant"
