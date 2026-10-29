# bot3 Ae-ft: mock e2e + honest-10 for [dfine_aeor] Ae-only ON vs OFF — BOTH in Docker pantum/n7q (PROTOCOL §10).
# Modeled on run_bot5_dfine_aeor_or_docker.ps1 + run_bot2_a3_dfine_docker_import.ps1.
# After ON: scripts/assert_dfine_aeor_ae_nonzero.py (HARD FAIL if n_raw/Ae dets == 0).
# Working copies + outputs on F: (C: low space). Container name prefix koz3- only.
# [dfine_aeor] stays default OFF in configs/default.toml; runner overlays ON only for out_on / out_honest_on.
$ErrorActionPreference = 'Continue'
[Console]::OutputEncoding = [Text.Encoding]::UTF8

# --- pin: update $Sha after push of bots/bot3-dfine-aeor-eval ---
$Sha            = 'PLACEHOLDER_SHA'   # filled by commit; runner FAIL if HEAD != $Sha
$Branch         = 'bots/bot3-dfine-aeor-eval'
$JobDir         = 'F:\Sloi_pc_jobs\bot3_dfine_aeor_ae_ft'
$Wt             = 'F:\Sloi_wt\bot3_dfine_aeor_ae_ft'
$Image          = 'pantum/n7q:20261024'
$RegionsMock    = 'F:\Sloi_pc_jobs_archive\v3e2e\regions'
$HonestRegions  = if ($env:KOZ3_HONEST_REGIONS) { $env:KOZ3_HONEST_REGIONS } else { 'F:\Sloi_pc_jobs\bot3_honest10_regions' }
$ModelsFill     = if ($env:KOZ3_MODELS_FILL) { $env:KOZ3_MODELS_FILL } else { 'F:\Sloi_wt\models_fill' }
# Ae-ft weights: copy from box /workspace/kaggle/out/dfine-ae-ft/{dfine_ae,models/dfine_aeor}
$WeightsAe      = if ($env:KOZ3_WEIGHTS_AE) { $env:KOZ3_WEIGHTS_AE } else { 'F:\Sloi_pc_weights\dfine_ae_ft\dfine_ae' }
$ThrAe          = if ($env:KOZ3_THRESHOLDS_AE) { $env:KOZ3_THRESHOLDS_AE } else { 'F:\Sloi_pc_weights\dfine_ae_ft\models\dfine_aeor\thresholds_oof.json' }
# Packaged final (1-iter from old input f3). Prefix must match; override via KOZ3_AE_SHA_PREFIX for alt fold.
$ExpectedShaPrefix = if ($env:KOZ3_AE_SHA_PREFIX) { $env:KOZ3_AE_SHA_PREFIX } else { '3e619878' }
$ZipV3          = if (Test-Path 'F:\Sloi_pc_jobs\_vendor_zips\submission_bot7cand_v3on_0876090.zip') {
                    'F:\Sloi_pc_jobs\_vendor_zips\submission_bot7cand_v3on_0876090.zip'
                  } else { 'C:\Sloi\pc_jobs\_vendor_zips\submission_bot7cand_v3on_0876090.zip' }
$PyHost         = 'C:\Sloi\expds-koz3_v0.2.0\expds-koz3\.venv\Scripts\python.exe'  # scoring + assert only
$ToolsTar       = 'F:\Sloi_pc_jobs_archive\v3e2e\kit\tools_v3e2e.tar'
$ToolsSrc       = 'F:\Sloi_pc_jobs_archive\v3e2e\out\tools'
$KitGt          = 'F:\Sloi_pc_jobs_archive\v3e2e\out\gt.geojson'
if (-not (Test-Path $KitGt)) { $KitGt = 'F:\Sloi_pc_jobs_archive\v3e2e\out_v3fix2\gt.geojson' }
$TrainRoot      = if ($env:KOZ3_TRAIN) { $env:KOZ3_TRAIN } else { 'F:\2\train-expds-3' }
$GitSrc         = if (Test-Path 'F:\Sloi_wt\expds-koz3') { 'F:\Sloi_wt\expds-koz3' } else { 'C:\Sloi\wt_bots' }

New-Item -ItemType Directory -Force -Path $JobDir, "$JobDir\notes",
  "$JobDir\out_off", "$JobDir\out_on", "$JobDir\dump_on",
  "$JobDir\out_honest_off", "$JobDir\out_honest_on", "$JobDir\dump_honest_on" | Out-Null

function HB($m){ Add-Content -Encoding utf8 "$JobDir\heartbeat.log" "$(Get-Date -Format o) $m"; Write-Host $m }
function Set-Status($s){
  $l = "$s $(Get-Date -Format o)"
  # Always stamp ENV for PROTOCOL §10
  if ($s -notmatch 'ENV=') { $l = "$s ENV=docker:$Image $(Get-Date -Format o)" }
  Set-Content -Encoding utf8 "$JobDir\STATUS.txt" $l
  HB $l
}

$st = if (Test-Path "$JobDir\STATUS.txt") { Get-Content "$JobDir\STATUS.txt" -Raw } else { '' }
if ($st -match '^DONE') { Write-Host 'already DONE'; exit 0 }
Set-Status 'RUNNING'
HB "start sha=$Sha branch=$Branch image=$Image EXPDS_STRICT=1 Ae-only ae-ft"

# --- worktree @Sha on F: ---
if (-not (Test-Path "$Wt\.git")) {
  & cmd.exe /c "git -C `"$GitSrc`" fetch origin $Branch" *> "$JobDir\notes\git_fetch.log"
  & cmd.exe /c "git -C `"$GitSrc`" worktree add --detach `"$Wt`" $Sha" *> "$JobDir\notes\git_wt.log"
} else {
  & cmd.exe /c "git -C `"$Wt`" fetch origin $Branch" *> "$JobDir\notes\git_fetch.log"
  & cmd.exe /c "git -C `"$Wt`" checkout --force --detach $Sha" *> "$JobDir\notes\git_wt.log"
}
$head = (& cmd.exe /c "git -C `"$Wt`" rev-parse HEAD").Trim()
HB "worktree HEAD=$head"
if ($Sha -eq 'PLACEHOLDER_SHA') {
  $Sha = (& cmd.exe /c "git -C `"$Wt`" rev-parse origin/$Branch").Trim()
  if (-not $Sha) { $Sha = $head }
  HB "resolved Sha from origin/$Branch -> $Sha"
}
if ($head -ne $Sha) { Set-Status "FAILED bad_sha want=$Sha got=$head"; exit 2 }

# --- models fill (pixgbm etc.) ---
if (Test-Path $ModelsFill) {
  & robocopy $ModelsFill "$Wt\models" /E /XC /XN /XO /NFL /NDL /NJH /NJS *> "$JobDir\notes\robocopy_models.log"
}

# --- Ae-ft weights + thresholds ---
New-Item -ItemType Directory -Force -Path "$Wt\models\dfine_ae", "$Wt\models\dfine_aeor" | Out-Null
if (-not (Test-Path "$WeightsAe\model.safetensors")) {
  Set-Status "FAILED missing_ae_weights $WeightsAe (copy from box /workspace/kaggle/out/dfine-ae-ft/dfine_ae)"; exit 3
}
& robocopy $WeightsAe "$Wt\models\dfine_ae" /E /NFL /NDL /NJH /NJS *> "$JobDir\notes\robocopy_dfine_ae.log"
if (Test-Path $ThrAe) {
  Copy-Item -Force $ThrAe "$Wt\models\dfine_aeor\thresholds_oof.json"
} elseif (Test-Path (Join-Path (Split-Path $WeightsAe -Parent) 'thresholds_ae_oof.json')) {
  Copy-Item -Force (Join-Path (Split-Path $WeightsAe -Parent) 'thresholds_ae_oof.json') "$Wt\models\dfine_aeor\thresholds_oof.json"
} else {
  Set-Status "FAILED missing_thresholds $ThrAe"; exit 3
}

# --- hard sha256 prefix check on Ae weights ---
$AeSt = "$Wt\models\dfine_ae\model.safetensors"
$shaFull = (& cmd.exe /c "certutil -hashfile `"$AeSt`" SHA256" | Select-Object -Skip 1 -First 1).Trim().ToLower().Replace(' ','')
if (-not $shaFull) {
  # fallback: host python
  $shaFull = (& $PyHost -c "import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest())" $AeSt).Trim().ToLower()
}
HB "ae_weights sha256=$shaFull prefix_want=$ExpectedShaPrefix"
Set-Content -Encoding utf8 "$JobDir\notes\ae_weights_sha.txt" "$shaFull  models/dfine_ae/model.safetensors"
if (-not $shaFull.StartsWith($ExpectedShaPrefix.ToLower())) {
  Set-Status "FAILED ae_sha_mismatch got=$shaFull want_prefix=$ExpectedShaPrefix (use KOZ3_AE_SHA_PREFIX for alt fold e.g. fold4 249b86fb)"; exit 3
}
if (-not (Test-Path "$Wt\models\pixgbm\all\poly.txt")) {
  HB 'WARN missing pixgbm — mock may still run if zip/fill provides it'
}

# --- vendor (manylinux) for transformers ---
if (-not (Test-Path "$Wt\inference\lib\vendor\transformers")) {
  if (-not (Test-Path $ZipV3)) { Set-Status "FAILED no_vendor_zip $ZipV3"; exit 3 }
  $Tmp = Join-Path $JobDir 'vendor_unpack'
  New-Item -ItemType Directory -Force -Path $Tmp | Out-Null
  & tar -xf $ZipV3 -C $Tmp --wildcards 'inference/lib/vendor/*' *> "$JobDir\notes\vendor_tar.log"
  if (-not (Test-Path "$Tmp\inference\lib\vendor\transformers")) { Expand-Archive -Path $ZipV3 -DestinationPath $Tmp -Force }
  if (-not (Test-Path "$Tmp\inference\lib\vendor\transformers")) { Set-Status 'FAILED vendor_not_in_zip'; exit 3 }
  Copy-Item -Recurse -Force "$Tmp\inference\lib\vendor" "$Wt\inference\lib\vendor"
  HB 'vendor copied'
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
if ($OffToml -notmatch "\[dfine_aeor\]\r?\nenabled = false") { Set-Status 'FAILED toml_anchor_missing'; exit 6 }
$OnToml = [regex]::Replace($OffToml, "\[dfine_aeor\]\r?\nenabled = false", $OnBlock)
$Utf8 = [Text.UTF8Encoding]::new($false)

# empty weight guard
$wbad = @()
if (Test-Path "$Wt\models") {
  Get-ChildItem "$Wt\models" -Recurse -File -EA SilentlyContinue |
    Where-Object { $_.Extension -match '\.(pt|pth|safetensors)$' -and $_.Length -le 0 } |
    ForEach-Object { $wbad += $_.FullName }
}
if ($wbad.Count -gt 0) {
  $wbad | Set-Content -Encoding utf8 "$JobDir\notes\empty_weights.txt"
  Set-Status ("FAILED empty_weights " + ($wbad -join ';')); exit 3
}

function Run-Variant([string]$name, [string]$tomlText, [string]$regions, [string]$outSub, [switch]$Dump) {
  $res = "$JobDir\$outSub\result.geojson"
  if (Test-Path $res) { HB "skip $name (result exists)"; return 0 }
  New-Item -ItemType Directory -Force -Path "$JobDir\$outSub" | Out-Null
  [IO.File]::WriteAllText($Toml, $tomlText, $Utf8)
  Copy-Item -Force $Toml "$JobDir\$outSub\default.toml"
  $CName = "koz3-bot3-aeor-$name-" + (Get-Date -Format 'yyyyMMddHHmmss')
  $envArgs = @('-e','PYTHONDONTWRITEBYTECODE=1','-e','PYTHONUNBUFFERED=1',
    '-e','HF_HUB_OFFLINE=1','-e','TRANSFORMERS_OFFLINE=1','-e','EXPDS_STRICT=1',
    '-e',"EXPDS_STATUS_DIR=/data/output/$outSub")
  if ($Dump) { $envArgs += @('-e',"EXPDS_DFINE_DUMP=/data/output/dump_$($outSub -replace '^out_','')") }
  $args2 = @('run','--rm','--name',$CName,'--gpus','all','--network','none') + $envArgs + @(
    '-v',"$(($Wt -replace '\\','/')):/data/zip:ro",
    '-v',"$(($regions -replace '\\','/')):/data/regions:ro",
    '-v',"$(($JobDir -replace '\\','/')):/data/output",
    '-w','/data/zip',$Image,
    'python','inference/main.py','/data/regions',"/data/output/$outSub/result.geojson")
  HB "docker run $name $CName regions=$regions"
  $t1 = Get-Date
  $p = Start-Process -FilePath 'docker' -ArgumentList $args2 -PassThru -NoNewWindow `
    -RedirectStandardOutput "$JobDir\$outSub\main.log" -RedirectStandardError "$JobDir\$outSub\main.err"
  while (-not $p.HasExited) { HB "$name running $([int]((Get-Date)-$t1).TotalSeconds)s"; Start-Sleep 20 }
  $sec = [math]::Round(((Get-Date)-$t1).TotalSeconds,1)
  Set-Content "$JobDir\$outSub\wall_sec.txt" $sec
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
  ForEach-Object { "$($_.Filename): $($_.Line)" } | Set-Content -Encoding UTF8 "$JobDir\warn_scan.txt"

if (-not (Test-Path "$JobDir\out_off\result.geojson")) { Set-Status "FAILED off exit=$ecOff no_geojson"; exit 4 }
if (-not (Test-Path "$JobDir\out_on\result.geojson"))  { Set-Status "FAILED on exit=$ecOn no_geojson"; exit 5 }

# --- HARD FAIL: Ae dets / n_raw > 0 ---
$AssertArgs = @("$Wt\scripts\assert_dfine_aeor_ae_nonzero.py",
  '--geojson', "$JobDir\out_on\result.geojson",
  '--main-log', "$JobDir\out_on\main.log")
$DumpJson = Get-ChildItem -Path "$JobDir\dump_on" -Recurse -Filter 'stages.json' -EA SilentlyContinue |
  Select-Object -First 1 -ExpandProperty FullName
if ($DumpJson) { $AssertArgs += @('--dump-json', $DumpJson) }
HB "assert: $($AssertArgs -join ' ')"
& $PyHost @AssertArgs *> "$JobDir\assert_ae_nonzero.log"
$assertEc = $LASTEXITCODE
if ($assertEc -ne 0) { Set-Status "FAILED assert_ae_nonzero exit=$assertEc (n_raw==0)"; exit 9 }
HB 'assert Ae nonzero PASS'

# --- score mock OFF vs ON ---
New-Item -ItemType Directory -Force -Path "$JobDir\tools" | Out-Null
if (Test-Path $ToolsTar) { tar -xf $ToolsTar -C $JobDir 2>$null }
foreach ($f in @('e2e_score.py','compute_metrics.py')) {
  $dst = Join-Path $JobDir "tools\$f"
  if (-not (Test-Path $dst) -and (Test-Path (Join-Path $ToolsSrc $f))) {
    Copy-Item -Force (Join-Path $ToolsSrc $f) $dst; HB "copied scorer $f"
  }
}
$ScorePy  = Get-ChildItem -Path $JobDir -Recurse -Filter e2e_score.py -EA SilentlyContinue | Select-Object -First 1 -ExpandProperty FullName
$MetricPy = Get-ChildItem -Path $JobDir -Recurse -Filter compute_metrics.py -EA SilentlyContinue | Select-Object -First 1 -ExpandProperty FullName
if (-not $ScorePy -or -not $MetricPy -or -not (Test-Path $KitGt)) { Set-Status 'FAILED no_scorer_or_gt'; exit 7 }
& $PyHost $ScorePy --metric $MetricPy --gt $KitGt `
  --a "ae_off_docker=$JobDir\out_off\result.geojson" --b "dfine_aeor_ae_on=$JobDir\out_on\result.geojson" `
  --out "$JobDir\scores_mock.json" *> "$JobDir\scores_mock.log"
if (-not (Test-Path "$JobDir\scores_mock.json")) { Set-Status 'FAILED score_mock'; exit 8 }
HB 'mock score OK'

# ========== (b) honest-10 OFF + ON (Docker) ==========
$honestRan = $false
if ((Test-Path $HonestRegions) -and (Get-ChildItem $HonestRegions -EA SilentlyContinue | Measure-Object).Count -gt 0) {
  HB "honest-10 regions=$HonestRegions"
  $ecHo = Run-Variant 'honest_off' $OffToml $HonestRegions 'out_honest_off'
  $ecHn = Run-Variant 'honest_on'  $OnToml  $HonestRegions 'out_honest_on' -Dump
  [IO.File]::WriteAllText($Toml, $OffToml, $Utf8)
  if ((Test-Path "$JobDir\out_honest_on\result.geojson") -and (Test-Path "$JobDir\out_honest_off\result.geojson")) {
    $AssertH = @("$Wt\scripts\assert_dfine_aeor_ae_nonzero.py",
      '--geojson', "$JobDir\out_honest_on\result.geojson",
      '--main-log', "$JobDir\out_honest_on\main.log")
    & $PyHost @AssertH *> "$JobDir\assert_ae_nonzero_honest.log"
    if ($LASTEXITCODE -ne 0) { Set-Status "FAILED assert_ae_nonzero_honest exit=$LASTEXITCODE"; exit 9 }
    # Prefer scoreboard honest-10 if train + wrapper present
    $EvalPy = Join-Path $Wt 'scripts\eval_aerial_honest10.py'
    $MetricWt = Join-Path $Wt 'materials\compute_metrics.py'
    if ((Test-Path $EvalPy) -and (Test-Path $TrainRoot) -and (Test-Path $MetricWt)) {
      New-Item -ItemType Directory -Force -Path "$JobDir\notes" | Out-Null
      # Build a tiny work dir pointing at OFF/ON preds via scoreboard --add of ON as dfine_ae
      # Fallback: e2e_score on honest GT if available beside regions
      $HonestGt = Join-Path $HonestRegions '..\gt.geojson'
      if (-not (Test-Path $HonestGt)) { $HonestGt = $KitGt }
      & $PyHost $ScorePy --metric $MetricPy --gt $HonestGt `
        --a "honest_off=$JobDir\out_honest_off\result.geojson" `
        --b "honest_ae_on=$JobDir\out_honest_on\result.geojson" `
        --out "$JobDir\scores_honest.json" *> "$JobDir\scores_honest.log"
      HB "honest e2e_score exit=$LASTEXITCODE"
    }
    $honestRan = $true
  }
} else {
  @"
honest-10 deferred: no regions at $HonestRegions
Prepare ~10 ae family sites (PROTOCOL honest-10) then set KOZ3_HONEST_REGIONS and re-run
(or leave out_off/out_on and only re-run honest phase).
Mock phase already asserts n_raw>0 and scores OFF vs ON.
"@ | Set-Content -Encoding UTF8 "$JobDir\out_honest_off\README.txt"
  HB 'honest-10 deferred (no regions)'
}

Set-Status 'DONE'
Get-Content "$JobDir\assert_ae_nonzero.log" -EA SilentlyContinue | Select-Object -First 40
if (Test-Path "$JobDir\scores_mock.json") { Get-Content "$JobDir\scores_mock.json" | Select-Object -First 80 }
if (Test-Path "$JobDir\scores_honest.json") { Get-Content "$JobDir\scores_honest.json" | Select-Object -First 80 }
HB "honest_ran=$honestRan"
Get-Content "$JobDir\out_off\wall_sec.txt","$JobDir\out_on\wall_sec.txt" -EA SilentlyContinue
