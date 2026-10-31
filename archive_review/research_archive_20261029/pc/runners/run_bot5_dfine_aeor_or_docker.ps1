# bot5 (c) mock e2e dfine_aeor Or-only ON vs flags-OFF v4 - BOTH in Docker pantum/n7q (PROTOCOL sect10).
# Modeled on run_bot2_a3_dfine_docker_import.ps1 + run_bot5_gor_dense_e2e_docker.ps1.
# After ON: scripts/assert_dfine_aeor_or_nonzero.py (HARD FAIL if n_raw/Or dets == 0).
# Container name prefix koz3- only. No leases (bot1 owns leases). Resumable via STATUS.txt + result.geojson.
$ErrorActionPreference = 'Continue'
[Console]::OutputEncoding = [Text.Encoding]::UTF8

$Sha       = '2b745df167d315de3a377ca112d1283454ab1953'
$JobDir    = 'C:\Sloi\pc_jobs\bot5_dfine_aeor_or_2b745df'
$Wt        = 'C:\Sloi\pc_jobs\wt_bot5_dfine_aeor_or'
$Image     = 'pantum/n7q:20261024'
$Regions   = 'F:\Sloi_pc_jobs_archive\v3e2e\regions'
$ModelsFill = 'C:\Sloi\wt_bot5_50f6f27\models'   # pixgbm etc.; Or folds may live here or under WeightsOr
$WeightsOr  = 'C:\Sloi\pc_weights\dfine_or'      # optional: models/dfine_or/dfine_f{0,1,2} layout
$ZipV3     = 'C:\Sloi\pc_jobs\_vendor_zips\submission_bot7cand_v3on_0876090.zip'
$PyHost    = 'C:\Sloi\expds-koz3_v0.2.0\expds-koz3\.venv\Scripts\python.exe'  # scoring + assert only
$ToolsTar  = 'F:\Sloi_pc_jobs_archive\v3e2e\kit\tools_v3e2e.tar'
$ToolsSrc  = 'F:\Sloi_pc_jobs_archive\v3e2e\out\tools'
$KitGt     = 'F:\Sloi_pc_jobs_archive\v3e2e\out\gt.geojson'

New-Item -ItemType Directory -Force -Path $JobDir, "$JobDir\notes", "$JobDir\out_off", "$JobDir\out_on", "$JobDir\dump_on" | Out-Null
function HB($m){ Add-Content -Encoding utf8 "$JobDir\heartbeat.log" "$(Get-Date -Format o) $m"; Write-Host $m }
function Set-Status($s){ $l="$s $(Get-Date -Format o)"; Set-Content -Encoding utf8 "$JobDir\STATUS.txt" $l; HB $l }
$st = if (Test-Path "$JobDir\STATUS.txt") { Get-Content "$JobDir\STATUS.txt" -Raw } else { '' }
if ($st -match '^DONE') { Write-Host 'already DONE'; exit 0 }
Set-Status 'RUNNING'
HB "start sha=$Sha image=$Image EXPDS_STRICT=1 Or-only"

# --- worktree @Sha ---
if (-not (Test-Path "$Wt\.git")) {
  & cmd.exe /c "git -C C:\Sloi\wt_bots fetch origin bots/bot5-candidate-v4" *> "$JobDir\notes\git_fetch.log"
  & cmd.exe /c "git -C C:\Sloi\wt_bots worktree add --detach `"$Wt`" $Sha" *> "$JobDir\notes\git_wt.log"
} else {
  & cmd.exe /c "git -C `"$Wt`" fetch origin bots/bot5-candidate-v4" *> "$JobDir\notes\git_fetch.log"
  & cmd.exe /c "git -C `"$Wt`" checkout --force --detach $Sha" *> "$JobDir\notes\git_wt.log"
}
$head = (& cmd.exe /c "git -C `"$Wt`" rev-parse HEAD").Trim()
HB "worktree HEAD=$head"
if ($head -ne $Sha) { Set-Status "FAILED bad_sha want=$Sha got=$head"; exit 2 }

# --- models fill (read-only source -> worktree); Or weights ---
if (Test-Path $ModelsFill) {
  & robocopy $ModelsFill "$Wt\models" /E /XC /XN /XO /NFL /NDL /NJH /NJS *> "$JobDir\notes\robocopy_models.log"
}
if (Test-Path $WeightsOr) {
  New-Item -ItemType Directory -Force -Path "$Wt\models\dfine_or" | Out-Null
  & robocopy $WeightsOr "$Wt\models\dfine_or" /E /NFL /NDL /NJH /NJS *> "$JobDir\notes\robocopy_dfine_or.log"
}
# Accept either models/dfine_or/dfine_f0 or models/dfine_or/f0 layout
$OrOk = $false
foreach ($cand in @(
  "$Wt\models\dfine_or\dfine_f0\model.safetensors",
  "$Wt\models\dfine_or\dfine_f0\config.json",
  "$Wt\models\dfine_or\f0\model.safetensors"
)) { if (Test-Path $cand) { $OrOk = $true; break } }
if (-not $OrOk) { Set-Status 'FAILED missing_dfine_or_weights'; exit 3 }
if (-not (Test-Path "$Wt\models\pixgbm\all\poly.txt")) { Set-Status 'FAILED missing_pixgbm'; exit 3 }

# --- vendor (manylinux) for transformers 5.21.0 ---
if (-not (Test-Path "$Wt\inference\lib\vendor\transformers")) {
  $Tmp = Join-Path $JobDir 'vendor_unpack'
  New-Item -ItemType Directory -Force -Path $Tmp | Out-Null
  & tar -xf $ZipV3 -C $Tmp --wildcards 'inference/lib/vendor/*' *> "$JobDir\notes\vendor_tar.log"
  if (-not (Test-Path "$Tmp\inference\lib\vendor\transformers")) { Expand-Archive -Path $ZipV3 -DestinationPath $Tmp -Force }
  if (-not (Test-Path "$Tmp\inference\lib\vendor\transformers")) { Set-Status 'FAILED vendor_not_in_zip'; exit 3 }
  Copy-Item -Recurse -Force "$Tmp\inference\lib\vendor" "$Wt\inference\lib\vendor"
  HB 'vendor copied'
}

# --- configs: OFF = default.toml; ON = Or-only dfine_aeor (Ae slot off / no ae weights) ---
$Toml = "$Wt\configs\default.toml"
$OffToml = [IO.File]::ReadAllText($Toml)
# Prefer dfine_f0/f1/f2 layout; runner notes alternate if only f0 present
$OrWeights = @('models/dfine_or/dfine_f0','models/dfine_or/dfine_f1','models/dfine_or/dfine_f2')
if (-not (Test-Path "$Wt\models\dfine_or\dfine_f0\model.safetensors")) {
  $OrWeights = @('models/dfine_or/f0','models/dfine_or/f1','models/dfine_or/f2')
}
$wList = ($OrWeights | ForEach-Object { "`"$_`"" }) -join ', '
$OnBlock = @(
  '[dfine_aeor]',
  'enabled = true',
  'markups = ["or"]',
  "weights = { or = [$wList] }",
  'thresholds_file = "models/dfine_aeor/thresholds_or_oof_provisional.json"',
  ''
) -join "`n"
if ($OffToml -notmatch "\[dfine_aeor\]\r?\nenabled = false") { Set-Status 'FAILED toml_anchor_missing'; exit 6 }
$OnToml = [regex]::Replace($OffToml, "\[dfine_aeor\]\r?\nenabled = false", $OnBlock)
$Utf8 = [Text.UTF8Encoding]::new($false)


# Preflight: reject empty weight files before docker (empty => torch.load EOFError)
$wbad = @()
if (Test-Path "$Wt\models") {
  Get-ChildItem "$Wt\models" -Recurse -File -EA SilentlyContinue |
    Where-Object { $_.Extension -match '\.(pt|pth|safetensors)$' -and $_.Length -le 0 } |
    ForEach-Object { $wbad += $_.FullName }
}
if ($wbad.Count -gt 0) {
  New-Item -ItemType Directory -Force -Path "$JobDir\notes" | Out-Null
  $wbad | Set-Content -Encoding utf8 "$JobDir\notes\empty_weights.txt"
  Set-Status ("FAILED empty_weights " + ($wbad -join ';')); exit 3
}

# Host torch.load smoke via temp .py file (PS5 strips quotes from -c args)
if (Test-Path $PyHost) {
  New-Item -ItemType Directory -Force -Path "$JobDir\notes" | Out-Null
  $crit = @(
    "$Wt\models\li_gor_mask_r18.pt",
    "$Wt\models\li_r18_hard2.pt",
    "$Wt\models\ae_fort_mask_r18.pt",
    "$Wt\models\spor_gor_r18.pt"
  ) | Where-Object { Test-Path $_ }
  if ($crit.Count -gt 0) {
    $py = @'
import sys, torch
bad=[]
for p in sys.argv[1:]:
  try:
    torch.load(p, map_location="cpu", weights_only=False)
  except Exception as e:
    bad.append("%s: %s" % (p, e))
if bad:
  print("FAIL"); [print(x) for x in bad]; sys.exit(1)
print("OK")
'@
    $pyf = Join-Path $JobDir "notes\torch_preflight.py"
    Set-Content -Encoding ascii -Path $pyf -Value $py
    $tr = & $PyHost $pyf @crit 2>&1
    HB "torch_preflight $tr"
    if ($LASTEXITCODE -ne 0) { Set-Status "FAILED torch_preflight"; exit 3 }
  }
}

function Run-Variant([string]$name, [string]$tomlText, [switch]$Dump) {
  $res = "$JobDir\out_$name\result.geojson"
  if (Test-Path $res) { HB "skip $name (result exists)"; return 0 }
  [IO.File]::WriteAllText($Toml, $tomlText, $Utf8)
  Copy-Item -Force $Toml "$JobDir\out_$name\default.toml"
  $CName = "koz3-bot5-aeor-$name-" + (Get-Date -Format 'yyyyMMddHHmmss')
  $envArgs = @('-e','PYTHONDONTWRITEBYTECODE=1','-e','PYTHONUNBUFFERED=1',
    '-e','HF_HUB_OFFLINE=1','-e','TRANSFORMERS_OFFLINE=1','-e','EXPDS_STRICT=1',
    '-e',"EXPDS_STATUS_DIR=/data/output/out_$name")
  if ($Dump) { $envArgs += @('-e','EXPDS_DFINE_DUMP=/data/output/dump_on') }
  $args2 = @('run','--rm','--name',$CName,'--gpus','all','--network','none') + $envArgs + @(
    '-v',"$(($Wt -replace '\\','/')):/data/zip:ro",
    '-v',"$(($Regions -replace '\\','/')):/data/regions:ro",
    '-v',"$(($JobDir -replace '\\','/')):/data/output",
    '-w','/data/zip',$Image,
    'python','inference/main.py','/data/regions',"/data/output/out_$name/result.geojson")
  HB "docker run $name $CName"
  $t1 = Get-Date
  $p = Start-Process -FilePath 'docker' -ArgumentList $args2 -PassThru -NoNewWindow `
    -RedirectStandardOutput "$JobDir\out_$name\main.log" -RedirectStandardError "$JobDir\out_$name\main.err"
  while (-not $p.HasExited) { HB "$name running $([int]((Get-Date)-$t1).TotalSeconds)s"; Start-Sleep 20 }
  $sec = [math]::Round(((Get-Date)-$t1).TotalSeconds,1)
  Set-Content "$JobDir\out_$name\wall_sec.txt" $sec
  HB "$name exit=$($p.ExitCode) wall_sec=$sec"
  return $p.ExitCode
}

$ecOff = Run-Variant 'off' $OffToml
$ecOn  = Run-Variant 'on'  $OnToml -Dump
[IO.File]::WriteAllText($Toml, $OffToml, $Utf8)   # restore

Select-String -Path "$JobDir\out_*\main.log","$JobDir\out_*\main.err" `
  -Pattern 'dfine_aeor|DFINE_AEOR|SP_DFINE_UNAVAILABLE|UNAVAILABLE|ImportError|Traceback|n_det=' -EA SilentlyContinue |
  ForEach-Object { "$($_.Filename): $($_.Line)" } | Set-Content -Encoding UTF8 "$JobDir\warn_scan.txt"

# docker Wait-Process exit=-1 quirk: salvage if geojson present
if (-not (Test-Path "$JobDir\out_off\result.geojson")) { Set-Status "FAILED off exit=$ecOff no_geojson"; exit 4 }
if (-not (Test-Path "$JobDir\out_on\result.geojson"))  { Set-Status "FAILED on exit=$ecOn no_geojson"; exit 5 }
if (($ecOff -ne 0 -and $ecOff -ne -1) -or ($ecOn -ne 0 -and $ecOn -ne -1)) {
  HB "WARN non-zero exit off=$ecOff on=$ecOn but geojson present - continue"
}

# --- HARD FAIL assert: Or dets / n_raw must be > 0 (not 'no change') ---
$AssertArgs = @("$Wt\scripts\assert_dfine_aeor_or_nonzero.py",
  '--geojson', "$JobDir\out_on\result.geojson",
  '--main-log', "$JobDir\out_on\main.log")
$DumpJson = Get-ChildItem -Path "$JobDir\dump_on" -Recurse -Filter 'stages.json' -EA SilentlyContinue |
  Select-Object -First 1 -ExpandProperty FullName
if ($DumpJson) { $AssertArgs += @('--dump-json', $DumpJson) }
HB "assert: $($AssertArgs -join ' ')"
& $PyHost @AssertArgs *> "$JobDir\assert_or_nonzero.log"
$assertEc = $LASTEXITCODE
if ($assertEc -ne 0) { Set-Status "FAILED assert_or_nonzero exit=$assertEc"; exit 9 }
HB 'assert Or nonzero PASS'

# --- score OFF vs ON (host python, scoring only) ---
# tools_v3e2e.tar lacks e2e_score.py; copy from archive out\tools
New-Item -ItemType Directory -Force -Path "$JobDir\tools" | Out-Null
if (Test-Path $ToolsTar) { tar -xf $ToolsTar -C $JobDir 2>$null }
foreach ($f in @('e2e_score.py','compute_metrics.py')) {
  $dst = Join-Path $JobDir "tools\$f"
  if (-not (Test-Path $dst) -and (Test-Path (Join-Path $ToolsSrc $f))) {
    Copy-Item -Force (Join-Path $ToolsSrc $f) $dst; HB "copied scorer $f from ToolsSrc"
  }
}
$ScorePy  = Get-ChildItem -Path $JobDir -Recurse -Filter e2e_score.py -EA SilentlyContinue | Select-Object -First 1 -ExpandProperty FullName
$MetricPy = Get-ChildItem -Path $JobDir -Recurse -Filter compute_metrics.py -EA SilentlyContinue | Select-Object -First 1 -ExpandProperty FullName
HB "scorer=$ScorePy metric=$MetricPy gt=$KitGt"
if (-not $ScorePy -or -not $MetricPy -or -not (Test-Path $KitGt)) { Set-Status 'FAILED no_scorer_or_gt'; exit 7 }
& $PyHost $ScorePy --metric $MetricPy --gt $KitGt `
  --a "v4off_docker=$JobDir\out_off\result.geojson" --b "dfine_aeor_or_on=$JobDir\out_on\result.geojson" `
  --out "$JobDir\scores.json" *> "$JobDir\scores.log"
if (-not (Test-Path "$JobDir\scores.json")) { Set-Status 'FAILED score'; exit 8 }

Set-Status 'DONE'
Get-Content "$JobDir\assert_or_nonzero.log" -EA SilentlyContinue | Select-Object -First 40
Get-Content "$JobDir\scores.json" | Select-Object -First 60
Get-Content "$JobDir\out_off\wall_sec.txt","$JobDir\out_on\wall_sec.txt" -EA SilentlyContinue
