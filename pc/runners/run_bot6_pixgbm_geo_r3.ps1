# bot-6 r3: PIXGBM REF vs NEW on KOZ-3 train geo regions 200-217 (not mock).
# Selective copy to C: (maggr/etgr/terfl + boundaries/noise/UTM only). Abort if plan >10 GB (exit 15).
# Never write to F:. Vendors geo_pixgbm.py (sha 35e4331b...). Gate: |dF1|<=0.01 ALL + maggr/etgr/terfl.
# Image pantum/n7q:20261024. Launch via schtasks (see launch_bot6_pixgbm_geo_r3_schtasks.ps1).
$ErrorActionPreference = 'Continue'
[Console]::OutputEncoding = [Text.Encoding]::UTF8

$Sha        = '13082a3e1f14be1590593f9fb887641cccb0484c'
$Branch     = 'bots/bot6-pixgbm-cleanroom'
$JobDir     = 'C:\Sloi\pc_jobs\bot6_pixgbm_geo_r3'
$Wt         = 'C:\Sloi\wt_bot6_pixgbm_geo_r3'
$Image      = 'pantum/n7q:20261024'
$TrainRoot  = 'F:\2\train-expds-3'
$RegionsDst = "$JobDir\regions"
$RefModels  = 'F:\Sloi_pc_jobs_archive\v3e2e\out_v3fix2\zip_v2\models\pixgbm\all'
$NewModels  = 'C:\Sloi\pc_jobs\bot6_pixgbm_b0add08\models\all'
$ModelsFill = 'C:\Sloi\wt_bot5_50f6f27\models'
$ZipV3      = 'C:\Sloi\pc_jobs\_vendor_zips\submission_bot7cand_v3on_0876090.zip'
$PyHost     = 'C:\Sloi\expds-koz3_v0.2.0\expds-koz3\.venv\Scripts\python.exe'
$ToolsTar   = 'F:\Sloi_pc_jobs_archive\v3e2e\kit\tools_v3e2e.tar'
$ToolsSrc   = 'F:\Sloi_pc_jobs_archive\v3e2e\out\tools'
$RefPolySha = 'd1fe3efafcd4377d41430d8828e881576c9024f9d7a5c89f55e23554e02ec800'
$RefFindsSha= '5fa8c09d77d0914cc925fb6bb925a0ab1b64ccecf3c823d7f3546e71c4ceb04a'
$GeoPixSha  = '35e4331bbc690e7eb3b31e07cc290837ce9fd96a05c92f4c999f24eceafda35b'
$GeoPixSrcRel = 'analysis\2026-10-29\bots\bot-4\runs\review\v2eb268d6\expds\geo_pixgbm.py'
$MaxCopyGb  = 10.0
$RunnersSrc = Split-Path -Parent $MyInvocation.MyCommand.Path
if (-not $RunnersSrc) { $RunnersSrc = 'C:\Sloi\expds-koz3\pc\runners' }

New-Item -ItemType Directory -Force -Path $JobDir, "$JobDir\notes", "$JobDir\out_ref", "$JobDir\out_new", "$JobDir\tools" | Out-Null
function HB($m){ Add-Content -Encoding utf8 "$JobDir\heartbeat.log" "$(Get-Date -Format o) $m"; Write-Host $m }
function Set-Status($s){ $l="$s $(Get-Date -Format o)"; Set-Content -Encoding utf8 "$JobDir\STATUS.txt" $l; HB $l }
$st = if (Test-Path "$JobDir\STATUS.txt") { Get-Content "$JobDir\STATUS.txt" -Raw } else { '' }
if ($st -match '^DONE') { Write-Host 'already DONE'; exit 0 }
Set-Status 'RUNNING'
HB "start sha=$Sha path=r3 geo_200_217 MaxCopyGb=$MaxCopyGb"

# Ensure helper scripts in JobDir
foreach ($h in @('bot6_r3_select_copy.py','bot6_r3_build_gt.py','bot6_r3_gate.py')) {
  $src = Join-Path $RunnersSrc $h
  if (-not (Test-Path $src)) { $src = Join-Path $JobDir $h }
  if (-not (Test-Path $src)) { Set-Status "FAILED missing_helper $h"; exit 6 }
  Copy-Item -Force $src (Join-Path $JobDir $h)
}

# worktree @ STRICT sha
if (-not (Test-Path "$Wt\.git")) {
  & cmd.exe /c "git -C C:\Sloi\wt_bots fetch origin $Branch" *> "$JobDir\notes\git_fetch.log"
  & cmd.exe /c "git -C C:\Sloi\wt_bots worktree add --detach `"$Wt`" $Sha" *> "$JobDir\notes\git_wt.log"
} else {
  & cmd.exe /c "git -C `"$Wt`" fetch origin $Branch" *> "$JobDir\notes\git_fetch.log"
  & cmd.exe /c "git -C `"$Wt`" checkout --force --detach $Sha" *> "$JobDir\notes\git_wt.log"
}
$head = (& cmd.exe /c "git -C `"$Wt`" rev-parse HEAD").Trim()
HB "worktree HEAD=$head"
if ($head -ne $Sha) { Set-Status "FAILED bad_sha want=$Sha got=$head"; exit 2 }

# models fill + vendor transformers
if (Test-Path $ModelsFill) {
  & robocopy $ModelsFill "$Wt\models" /E /XC /XN /XO /NFL /NDL /NJH /NJS *> "$JobDir\notes\robocopy_models.log"
}
if (-not (Test-Path "$Wt\inference\lib\vendor\transformers") -and (Test-Path $ZipV3)) {
  $Tmp = Join-Path $JobDir 'vendor_unpack'
  New-Item -ItemType Directory -Force -Path $Tmp | Out-Null
  & tar -xf $ZipV3 -C $Tmp --wildcards 'inference/lib/vendor/*' *> "$JobDir\notes\vendor_tar.log"
  if (-not (Test-Path "$Tmp\inference\lib\vendor\transformers")) { Expand-Archive -Path $ZipV3 -DestinationPath $Tmp -Force }
  if (Test-Path "$Tmp\inference\lib\vendor\transformers") {
    Copy-Item -Recurse -Force "$Tmp\inference\lib\vendor" "$Wt\inference\lib\vendor"
    HB 'vendor copied'
  }
}

# vendor expds/geo_pixgbm.py (gitignored)
$GeoPixDst = "$Wt\expds\geo_pixgbm.py"
$GeoPixSrc = Join-Path $Wt $GeoPixSrcRel
if (-not (Test-Path $GeoPixSrc)) { Set-Status "FAILED missing_geo_pixgbm_src $GeoPixSrc"; exit 6 }
Copy-Item -Force $GeoPixSrc $GeoPixDst
$gp = (Get-FileHash -Algorithm SHA256 $GeoPixDst).Hash.ToLower()
HB "geo_pixgbm.py vendored sha=$gp"
if ($gp -ne $GeoPixSha) { Set-Status "FAILED geo_pixgbm_sha $gp"; exit 6 }

# REF / NEW models
foreach ($f in @("$NewModels\poly.txt","$NewModels\finds.txt")) {
  if (-not (Test-Path $f)) { Set-Status "FAILED missing_new $f"; exit 3 }
}
$refOk = (Test-Path "$RefModels\poly.txt")
if (-not $refOk) {
  foreach ($cand in @(
    'F:\Sloi_pc_jobs_archive\v3e2e\out_v3fix2\zip_v3\models\pixgbm\all',
    "$Wt\models\pixgbm\all",
    'C:\Sloi\wt_bot6_b0add08\models\pixgbm\all'
  )) {
    if (Test-Path "$cand\poly.txt") { $RefModels = $cand; $refOk = $true; break }
  }
}
if (-not $refOk) { Set-Status "FAILED missing_ref_models"; exit 3 }
HB "RefModels=$RefModels NewModels=$NewModels"
$rp = (Get-FileHash -Algorithm SHA256 "$RefModels\poly.txt").Hash.ToLower()
$rf = (Get-FileHash -Algorithm SHA256 "$RefModels\finds.txt").Hash.ToLower()
$np = (Get-FileHash -Algorithm SHA256 "$NewModels\poly.txt").Hash.ToLower()
$nf = (Get-FileHash -Algorithm SHA256 "$NewModels\finds.txt").Hash.ToLower()
HB "ref poly=$rp finds=$rf new poly=$np finds=$nf"
if ($rp -ne $RefPolySha -or $rf -ne $RefFindsSha) {
  HB "WARN ref sha != MODELS_NOT_IN_GIT - continue with documented REF path"
}
Set-Content -Encoding utf8 "$JobDir\notes\input_hashes.txt" @"
ref_poly $rp $RefModels\poly.txt
ref_finds $rf $RefModels\finds.txt
new_poly $np $NewModels\poly.txt
new_finds $nf $NewModels\finds.txt
sha $Sha
"@

# ---- Selective geo copy (plan first; abort exit 15 if > MaxCopyGb) ----
if (-not (Test-Path $PyHost)) { Set-Status "FAILED no_PyHost $PyHost"; exit 6 }
if (-not (Test-Path $TrainRoot)) { Set-Status "FAILED train_root_missing $TrainRoot"; exit 16 }
$PlanOut = "$JobDir\notes\copy_plan.json"
HB "select_copy PLAN root=$TrainRoot dst=$RegionsDst max_gb=$MaxCopyGb"
& $PyHost "$JobDir\bot6_r3_select_copy.py" --repo $Wt --root $TrainRoot --dst $RegionsDst --plan-out $PlanOut --max-gb $MaxCopyGb *> "$JobDir\notes\select_copy_plan.log"
$planEc = $LASTEXITCODE
HB "select_copy plan exit=$planEc"
if (Test-Path $PlanOut) {
  $planTxt = Get-Content $PlanOut -Raw -Encoding UTF8
  HB "copy_plan summary bytes/gb in notes\copy_plan.json"
  Add-Content -Encoding utf8 "$JobDir\notes\copy_size.txt" $planTxt
}
if ($planEc -eq 15) { Set-Status "FAILED copy_would_exceed_max_gb=$MaxCopyGb (exit 15)"; exit 15 }
if ($planEc -ne 0) { Set-Status "FAILED select_copy_plan exit=$planEc"; exit $planEc }

# Parse total_gb from plan for heartbeat
$CopyGb = '?'
try {
  $pj = Get-Content $PlanOut -Raw -Encoding UTF8 | ConvertFrom-Json
  $CopyGb = $pj.total_gb
  HB "planned_copy_gb=$CopyGb n_files=$($pj.n_files) n_regions=$($pj.n_regions)"
  Set-Content -Encoding utf8 "$JobDir\notes\planned_copy_gb.txt" "planned_gb=$CopyGb max_gb=$MaxCopyGb n_files=$($pj.n_files) n_regions=$($pj.n_regions)"
} catch { HB "WARN could not parse plan JSON" }

if (-not (Test-Path "$RegionsDst\_region_list.txt")) {
  HB "select_copy COPY starting"
  & $PyHost "$JobDir\bot6_r3_select_copy.py" --repo $Wt --root $TrainRoot --dst $RegionsDst --plan-out $PlanOut --max-gb $MaxCopyGb --copy *> "$JobDir\notes\select_copy_do.log"
  $copyEc = $LASTEXITCODE
  HB "select_copy copy exit=$copyEc"
  if ($copyEc -eq 15) { Set-Status "FAILED copy_would_exceed_max_gb=$MaxCopyGb"; exit 15 }
  if ($copyEc -ne 0) { Set-Status "FAILED select_copy exit=$copyEc"; exit $copyEc }
} else {
  HB "regions already present (region_list exists) - skip copy"
}

# ---- Build GT (geo markups only; annotations read from F:, write to C:) ----
$GtOut = "$JobDir\gt.geojson"
if (-not (Test-Path $GtOut)) {
  HB "build_gt from $TrainRoot"
  & $PyHost "$JobDir\bot6_r3_build_gt.py" --repo $Wt --root $TrainRoot --out $GtOut --regions-from "$RegionsDst\_region_list.txt" *> "$JobDir\notes\build_gt.log"
  if ($LASTEXITCODE -ne 0 -or -not (Test-Path $GtOut)) { Set-Status "FAILED build_gt exit=$LASTEXITCODE"; exit 18 }
} else { HB "gt.geojson exists - skip" }

# ---- Patch toml: pixgbm ON + large budget + watchdog false ----
$Toml = "$Wt\configs\default.toml"
$BaseToml = [IO.File]::ReadAllText($Toml)
$OnToml = $BaseToml
$OnToml = [regex]::Replace($OnToml, '\[geophys\.pixgbm\]\r?\nenabled\s*=\s*false', "[geophys.pixgbm]`nenabled = true")
$OnToml = [regex]::Replace($OnToml, '(?m)^replace\s*=\s*false', 'replace = true')
# Raise ALL budget_s (time + pixgbm) and disable watchdog (geo pack can exceed 1500s)
$OnToml = [regex]::Replace($OnToml, '(?m)^budget_s\s*=\s*\d+', 'budget_s = 100000')
$OnToml = [regex]::Replace($OnToml, '(?m)^watchdog\s*=\s*true', 'watchdog = false')
$Utf8 = [Text.UTF8Encoding]::new($false)
Copy-Item -Force $Toml "$JobDir\notes\default.toml.bak"

function Install-PixModels([string]$src) {
  $dst = "$Wt\models\pixgbm\all"
  New-Item -ItemType Directory -Force -Path $dst | Out-Null
  Copy-Item -Force "$src\poly.txt" "$dst\poly.txt"
  Copy-Item -Force "$src\finds.txt" "$dst\finds.txt"
}

# empty-weight preflight
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

# Host torch preflight via temp .py (no python -c)
if (Test-Path $PyHost) {
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

# Meta accumulators for gate
$Meta = @{
  docker_ec = @{}; t_start_epoch = @{}; incontainer_ok = @{}
  result_identical = $null; model_sha_differ = ($rp -ne $np -or $rf -ne $nf)
  result_sha = @{}; planned_copy_gb = $CopyGb
}

function Run-Variant([string]$name, [string]$srcModels) {
  $res = "$JobDir\out_$name\result.geojson"
  if (Test-Path $res) { HB "skip $name (result exists)"; $Meta.docker_ec[$name] = 0; if (-not $Meta.t_start_epoch[$name]) { $Meta.t_start_epoch[$name] = 0 }; return 0 }
  Install-PixModels $srcModels
  [IO.File]::WriteAllText($Toml, $OnToml, $Utf8)
  Copy-Item -Force $Toml "$JobDir\out_$name\default.toml"
  Copy-Item -Force "$Wt\models\pixgbm\all\poly.txt" "$JobDir\out_$name\poly.txt"
  Copy-Item -Force "$Wt\models\pixgbm\all\finds.txt" "$JobDir\out_$name\finds.txt"
  $CName = "koz3-bot6-geo-r3-$name-" + (Get-Date -Format 'yyyyMMddHHmmss')
  $tEpoch = [DateTimeOffset]::UtcNow.ToUnixTimeSeconds()
  $Meta.t_start_epoch[$name] = $tEpoch
  $args2 = @(
    'run','--rm','--name',$CName,'--gpus','all','--network','none',
    '-e','PYTHONDONTWRITEBYTECODE=1','-e','PYTHONUNBUFFERED=1',
    '-e','HF_HUB_OFFLINE=1','-e','TRANSFORMERS_OFFLINE=1',
    '-e','EXPDS_PIXGBM_STRICT=1','-e','PIXGBM_BUDGET_S=100000','-e','EXPDS_SOFT_CEILING_S=0',
    '-v',"$(($Wt -replace '\\','/')):/data/zip:ro",
    '-v',"$(($RegionsDst -replace '\\','/')):/data/regions:ro",
    '-v',"$(($JobDir -replace '\\','/')):/data/output",
    '-w','/data/zip',$Image,
    'sh','-c',"`"sha256sum models/pixgbm/all/poly.txt models/pixgbm/all/finds.txt expds/geo_pixgbm.py > /data/output/out_$name/incontainer_sha.txt 2>&1; python inference/main.py /data/regions /data/output/out_$name/result.geojson`""
  )
  HB "docker run $name $CName STRICT=1 BUDGET=100000 soft_ceiling=0"
  $t1 = Get-Date
  $p = Start-Process -FilePath 'docker' -ArgumentList $args2 -PassThru -NoNewWindow `
    -RedirectStandardOutput "$JobDir\out_$name\main.log" -RedirectStandardError "$JobDir\out_$name\main.err"
  while (-not $p.HasExited) { HB "$name running $([int]((Get-Date)-$t1).TotalSeconds)s"; Start-Sleep 30 }
  $sec = [math]::Round(((Get-Date)-$t1).TotalSeconds,1)
  Set-Content "$JobDir\out_$name\wall_sec.txt" $sec
  $Meta.docker_ec[$name] = $p.ExitCode
  HB "$name exit=$($p.ExitCode) wall_sec=$sec"
  return $p.ExitCode
}

$ecRef = Run-Variant 'ref' $RefModels
$ecNew = Run-Variant 'new' $NewModels
[IO.File]::WriteAllText($Toml, $BaseToml, $Utf8)

if (-not (Test-Path "$JobDir\out_ref\result.geojson")) { Set-Status "FAILED ref exit=$ecRef no_geojson"; exit 4 }
if (-not (Test-Path "$JobDir\out_new\result.geojson")) { Set-Status "FAILED new exit=$ecNew no_geojson"; exit 5 }

# in-container sha check
$want = @{ ref = @($rp,$rf); new = @($np,$nf) }
foreach ($v in 'ref','new') {
  $shaTxt = Get-Content "$JobDir\out_$v\incontainer_sha.txt" -Raw -EA SilentlyContinue
  HB "incontainer_sha $v :: $($shaTxt -replace "`r?`n",' | ')"
  $ok = $shaTxt -and ($shaTxt -match $want[$v][0]) -and ($shaTxt -match $want[$v][1]) -and ($shaTxt -match $GeoPixSha)
  $Meta.incontainer_ok[$v] = [bool]$ok
  if (-not $ok) { Set-Status "FAILED incontainer_sha_mismatch $v"; exit 10 }
}

$hr = (Get-FileHash -Algorithm SHA256 "$JobDir\out_ref\result.geojson").Hash.ToLower()
$hn = (Get-FileHash -Algorithm SHA256 "$JobDir\out_new\result.geojson").Hash.ToLower()
$Meta.result_identical = ($hr -eq $hn)
$Meta.result_sha = @{ ref = $hr; new = $hn }
HB "result.geojson sha ref=$hr new=$hn identical=$($hr -eq $hn)"
$metaPy = @'
import json, sys
from pathlib import Path
job = Path(sys.argv[1])
meta = {
  "docker_ec": {"ref": int(sys.argv[2]), "new": int(sys.argv[3])},
  "t_start_epoch": {"ref": int(sys.argv[4]), "new": int(sys.argv[5])},
  "incontainer_ok": {"ref": sys.argv[6] == "1", "new": sys.argv[7] == "1"},
  "result_identical": sys.argv[8] == "1",
  "model_sha_differ": sys.argv[9] == "1",
  "result_sha": {"ref": sys.argv[10], "new": sys.argv[11]},
  "planned_copy_gb": sys.argv[12],
}
(job / "notes" / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
print(json.dumps(meta))
'@
$metaPyFile = Join-Path $JobDir 'notes\write_run_meta.py'
Set-Content -Encoding ascii -Path $metaPyFile -Value $metaPy
$ident = if ($hr -eq $hn) { '1' } else { '0' }
$mdiff = if ($rp -ne $np -or $rf -ne $nf) { '1' } else { '0' }
$okR = if ($Meta.incontainer_ok['ref']) { '1' } else { '0' }
$okN = if ($Meta.incontainer_ok['new']) { '1' } else { '0' }
$tsR = $Meta.t_start_epoch['ref']; if (-not $tsR) { $tsR = 0 }
$tsN = $Meta.t_start_epoch['new']; if (-not $tsN) { $tsN = 0 }
$ecR = $Meta.docker_ec['ref']; if ($null -eq $ecR) { $ecR = -1 }
$ecN = $Meta.docker_ec['new']; if ($null -eq $ecN) { $ecN = -1 }
& $PyHost $metaPyFile $JobDir $ecR $ecN $tsR $tsN $okR $okN $ident $mdiff $hr $hn "$CopyGb"

# score
if (Test-Path $ToolsTar) { tar -xf $ToolsTar -C $JobDir 2>$null }
foreach ($f in @('e2e_score.py','compute_metrics.py')) {
  $dst = Join-Path $JobDir "tools\$f"
  if (-not (Test-Path $dst) -and (Test-Path (Join-Path $ToolsSrc $f))) {
    Copy-Item -Force (Join-Path $ToolsSrc $f) $dst; HB "copied scorer $f"
  }
}
$ScorePy  = Get-ChildItem -Path $JobDir -Recurse -Filter e2e_score.py -EA SilentlyContinue | Select-Object -First 1 -ExpandProperty FullName
$MetricPy = Get-ChildItem -Path $JobDir -Recurse -Filter compute_metrics.py -EA SilentlyContinue | Select-Object -First 1 -ExpandProperty FullName
HB "scorer=$ScorePy metric=$MetricPy gt=$GtOut"
if (-not $ScorePy -or -not $MetricPy -or -not (Test-Path $GtOut)) { Set-Status 'FAILED no_scorer_or_gt'; exit 7 }
& $PyHost $ScorePy --metric $MetricPy --gt $GtOut `
  --a "pix_ref=$JobDir\out_ref\result.geojson" --b "pix_new=$JobDir\out_new\result.geojson" `
  --out "$JobDir\scores.json" *> "$JobDir\scores.log"
if (-not (Test-Path "$JobDir\scores.json")) { Set-Status 'FAILED score'; exit 8 }

# gate (validity + |dF1|<=0.01)
& $PyHost "$JobDir\bot6_r3_gate.py" --job $JobDir --scores "$JobDir\scores.json" *> "$JobDir\notes\gate_run.log"
$gateEc = $LASTEXITCODE
HB "gate exit=$gateEc"
if ($gateEc -ne 0) {
  Set-Status "FAILED gate exit=$gateEc"
  if (Test-Path "$JobDir\gate.json") { Get-Content "$JobDir\gate.json" }
  exit $gateEc
}
Set-Status 'DONE'
Get-Content "$JobDir\gate.json" -EA SilentlyContinue
Get-Content "$JobDir\scores.json" -EA SilentlyContinue | Select-Object -First 80
