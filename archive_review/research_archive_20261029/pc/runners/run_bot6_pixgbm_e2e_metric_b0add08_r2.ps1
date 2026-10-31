# bot-6 path (b) r2: mock e2e PIXGBM S REF vs NEW @b0add08. r1 was FALSE-EQUAL: expds/geo_pixgbm.py is .gitignored/untracked
# -> git worktree lacks it -> pipeline ImportError -> classic in BOTH runs. r2 vendors geo_pixgbm.py (sha 35e4331b, = v2 eb268d6)
# from the tracked copy in analysis/, logs in-container sha, and hard-fails unless pix actually ran.
# Gate: DeltaALL(F1_new-F1_ref)>=-0.01; no layer drop >0.01. Image pantum/n7q:20261024.
$ErrorActionPreference = 'Continue'
[Console]::OutputEncoding = [Text.Encoding]::UTF8

$Sha        = 'b0add08e194853a351d520df19cfb9810c4db983'
$Branch     = 'bots/bot6-pixgbm-cleanroom'
$JobDir     = 'C:\Sloi\pc_jobs\bot6_pixgbm_e2e_metric_b0add08_r2'
$Wt         = 'C:\Sloi\wt_bot6_e2e_metric_b0add08'
$Image      = 'pantum/n7q:20261024'
$Regions    = 'F:\Sloi_pc_jobs_archive\v3e2e\regions'
$RefModels  = 'F:\Sloi_pc_jobs_archive\v3e2e\out_v3fix2\zip_v2\models\pixgbm\all'
$NewModels  = 'C:\Sloi\pc_jobs\bot6_pixgbm_b0add08\models\all'
$ModelsFill = 'C:\Sloi\wt_bot5_50f6f27\models'
$ZipV3      = 'C:\Sloi\pc_jobs\_vendor_zips\submission_bot7cand_v3on_0876090.zip'
$PyHost     = 'C:\Sloi\expds-koz3_v0.2.0\expds-koz3\.venv\Scripts\python.exe'
$ToolsTar   = 'F:\Sloi_pc_jobs_archive\v3e2e\kit\tools_v3e2e.tar'
$ToolsSrc   = 'F:\Sloi_pc_jobs_archive\v3e2e\out\tools'
$KitGt      = 'F:\Sloi_pc_jobs_archive\v3e2e\out\gt.geojson'
$RefPolySha = 'd1fe3efafcd4377d41430d8828e881576c9024f9d7a5c89f55e23554e02ec800'
$RefFindsSha= '5fa8c09d77d0914cc925fb6bb925a0ab1b64ccecf3c823d7f3546e71c4ceb04a'
$GeoPixSha  = '35e4331bbc690e7eb3b31e07cc290837ce9fd96a05c92f4c999f24eceafda35b'
$GeoPixSrcRel = 'analysis\2026-10-29\bots\bot-4\runs\review\v2eb268d6\expds\geo_pixgbm.py'

New-Item -ItemType Directory -Force -Path $JobDir, "$JobDir\notes", "$JobDir\out_ref", "$JobDir\out_new" | Out-Null
function HB($m){ Add-Content -Encoding utf8 "$JobDir\heartbeat.log" "$(Get-Date -Format o) $m"; Write-Host $m }
function Set-Status($s){ $l="$s $(Get-Date -Format o)"; Set-Content -Encoding utf8 "$JobDir\STATUS.txt" $l; HB $l }
$st = if (Test-Path "$JobDir\STATUS.txt") { Get-Content "$JobDir\STATUS.txt" -Raw } else { '' }
if ($st -match '^DONE') { Write-Host 'already DONE'; exit 0 }
Set-Status 'RUNNING'
HB "start sha=$Sha path=b e2e-metric"

# worktree
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

# models fill (missing only) + vendor
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

# vendor expds/geo_pixgbm.py (gitignored -> absent in worktree; without it pipeline silently falls back to classic)
$GeoPixDst = "$Wt\expds\geo_pixgbm.py"
$GeoPixSrc = Join-Path $Wt $GeoPixSrcRel
if (-not (Test-Path $GeoPixSrc)) { Set-Status "FAILED missing_geo_pixgbm_src $GeoPixSrc"; exit 6 }
Copy-Item -Force $GeoPixSrc $GeoPixDst
$gp = (Get-FileHash -Algorithm SHA256 $GeoPixDst).Hash.ToLower()
HB "geo_pixgbm.py vendored sha=$gp"
if ($gp -ne $GeoPixSha) { Set-Status "FAILED geo_pixgbm_sha $gp"; exit 6 }

# verify NEW models exist
foreach ($f in @("$NewModels\poly.txt","$NewModels\finds.txt")) {
  if (-not (Test-Path $f)) { Set-Status "FAILED missing_new $f"; exit 3 }
}
# REF fallbacks
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

# sha check REF
$rp = (Get-FileHash -Algorithm SHA256 "$RefModels\poly.txt").Hash.ToLower()
$rf = (Get-FileHash -Algorithm SHA256 "$RefModels\finds.txt").Hash.ToLower()
HB "ref poly=$rp finds=$rf"
if ($rp -ne $RefPolySha -or $rf -ne $RefFindsSha) {
  HB "WARN ref sha != MODELS_NOT_IN_GIT (poly want $RefPolySha finds want $RefFindsSha) - continue with documented REF path"
}
Set-Content -Encoding utf8 "$JobDir\notes\input_hashes.txt" @"
ref_poly $rp $RefModels\poly.txt
ref_finds $rf $RefModels\finds.txt
new_poly $((Get-FileHash -Algorithm SHA256 "$NewModels\poly.txt").Hash.ToLower()) $NewModels\poly.txt
new_finds $((Get-FileHash -Algorithm SHA256 "$NewModels\finds.txt").Hash.ToLower()) $NewModels\finds.txt
sha $Sha
"@

# enable pixgbm S in toml (flag OFF default -> ON for this job only)
$Toml = "$Wt\configs\default.toml"
$BaseToml = [IO.File]::ReadAllText($Toml)
if ($BaseToml -notmatch '\[geophys\.pixgbm\]') { Set-Status 'FAILED no [geophys.pixgbm] in default.toml'; exit 6 }
$PixOn = @'
[geophys.pixgbm]
enabled = true
replace = true
model_dir = "models/pixgbm/all"
classes = ["finds_points", "object_poly"]
'@
# Replace enabled=false block start - keep other keys after by only flipping enabled/replace
$OnToml = [regex]::Replace($BaseToml, '\[geophys\.pixgbm\]\r?\nenabled\s*=\s*false', "[geophys.pixgbm]`nenabled = true")
$OnToml = [regex]::Replace($OnToml, '(?m)^replace\s*=\s*false', 'replace = true')
$Utf8 = [Text.UTF8Encoding]::new($false)

function Install-PixModels([string]$src) {
  $dst = "$Wt\models\pixgbm\all"
  New-Item -ItemType Directory -Force -Path $dst | Out-Null
  Copy-Item -Force "$src\poly.txt" "$dst\poly.txt"
  Copy-Item -Force "$src\finds.txt" "$dst\finds.txt"
}


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

function Run-Variant([string]$name, [string]$srcModels) {
  $res = "$JobDir\out_$name\result.geojson"
  if (Test-Path $res) { HB "skip $name (result exists)"; return 0 }
  Install-PixModels $srcModels
  [IO.File]::WriteAllText($Toml, $OnToml, $Utf8)
  Copy-Item -Force $Toml "$JobDir\out_$name\default.toml"
  Copy-Item -Force "$Wt\models\pixgbm\all\poly.txt" "$JobDir\out_$name\poly.txt"
  Copy-Item -Force "$Wt\models\pixgbm\all\finds.txt" "$JobDir\out_$name\finds.txt"
  $CName = "koz3-bot6-pix-$name-" + (Get-Date -Format 'yyyyMMddHHmmss')
  $args2 = @('run','--rm','--name',$CName,'--gpus','all','--network','none',
    '-e','PYTHONDONTWRITEBYTECODE=1','-e','PYTHONUNBUFFERED=1','-e','HF_HUB_OFFLINE=1','-e','TRANSFORMERS_OFFLINE=1',
    '-v',"$(($Wt -replace '\\','/')):/data/zip:ro",'-v',"$(($Regions -replace '\\','/')):/data/regions:ro",
    '-v',"$(($JobDir -replace '\\','/')):/data/output",'-w','/data/zip',$Image,
    'sh','-c',"`"sha256sum models/pixgbm/all/poly.txt models/pixgbm/all/finds.txt expds/geo_pixgbm.py > /data/output/out_$name/incontainer_sha.txt 2>&1; python inference/main.py /data/regions /data/output/out_$name/result.geojson`"")
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

$ecRef = Run-Variant 'ref' $RefModels
$ecNew = Run-Variant 'new' $NewModels
[IO.File]::WriteAllText($Toml, $BaseToml, $Utf8)  # restore OFF

if (-not (Test-Path "$JobDir\out_ref\result.geojson")) { Set-Status "FAILED ref exit=$ecRef no_geojson"; exit 4 }
if (-not (Test-Path "$JobDir\out_new\result.geojson")) { Set-Status "FAILED new exit=$ecNew no_geojson"; exit 5 }
if (($ecRef -ne 0 -and $ecRef -ne -1) -or ($ecNew -ne 0 -and $ecNew -ne -1)) {
  HB "WARN non-zero exit ref=$ecRef new=$ecNew but geojson present - continue score"
}

# proof pix actually ran + correct models mounted (r1 bug guard)
$want = @{ ref = @($rp,$rf); new = @((Get-FileHash -Algorithm SHA256 "$NewModels\poly.txt").Hash.ToLower(), (Get-FileHash -Algorithm SHA256 "$NewModels\finds.txt").Hash.ToLower()) }
foreach ($v in 'ref','new') {
  $shaTxt = Get-Content "$JobDir\out_$v\incontainer_sha.txt" -Raw -EA SilentlyContinue
  HB "incontainer_sha $v :: $($shaTxt -replace "`r?`n",' | ')"
  if (-not $shaTxt -or $shaTxt -notmatch $want[$v][0] -or $shaTxt -notmatch $want[$v][1] -or $shaTxt -notmatch $GeoPixSha) {
    Set-Status "FAILED incontainer_sha_mismatch $v"; exit 10 }
  $logTxt = ((Get-Content "$JobDir\out_$v\main.log" -Raw -Encoding UTF8 -EA SilentlyContinue) + (Get-Content "$JobDir\out_$v\main.err" -Raw -Encoding UTF8 -EA SilentlyContinue))
  $pixLines = ([regex]::Matches($logTxt, 'pixgbm [^\r\n]*backend=[^\r\n]*')).Value
  $pixLines | Set-Content -Encoding utf8 "$JobDir\out_$v\pix_lines.txt"
  $bad = ([regex]::Matches($logTxt, '[^\r\n]*(нет в этой сборке|pixgbm пропущен(?! —)|нет poly\.txt)[^\r\n]*')).Value
  HB "pix $v lines=$($pixLines.Count) full_hits=$(($pixLines | Select-String "'full'").Count) bad=$($bad.Count)"
  if ($bad.Count -gt 0) { $bad | Set-Content -Encoding utf8 "$JobDir\out_$v\pix_bad.txt"; Set-Status "FAILED pix_not_run $v (see out_$v\pix_bad.txt)"; exit 11 }
  if (($pixLines | Select-String "'full'|'finds_only'").Count -eq 0) { Set-Status "FAILED pix_no_full_step $v"; exit 11 }
}
$hr = (Get-FileHash -Algorithm SHA256 "$JobDir\out_ref\result.geojson").Hash
$hn = (Get-FileHash -Algorithm SHA256 "$JobDir\out_new\result.geojson").Hash
HB "result.geojson sha ref=$hr new=$hn identical=$($hr -eq $hn)"
Set-Content -Encoding utf8 "$JobDir\notes\result_identical.txt" "ref=$hr new=$hn identical=$($hr -eq $hn)"

# score
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
  --a "pix_ref=$JobDir\out_ref\result.geojson" --b "pix_new=$JobDir\out_new\result.geojson" `
  --out "$JobDir\scores.json" *> "$JobDir\scores.log"
if (-not (Test-Path "$JobDir\scores.json")) { Set-Status 'FAILED score'; exit 8 }

# gate: DeltaALL >= -0.01; no layer drop > 0.01 (parsed by host python)
$gatePy = @'
import json, sys
from pathlib import Path
job = Path(sys.argv[1])
d = json.loads((job / "scores.json").read_text(encoding="utf-8"))
rows = d.get("rows") or []
gate = {"pass": False, "reason": "unparsed", "tol": -0.01, "layer_tol": 0.01}
allrow = next((r for r in rows if r.get("region") == "ALL" and r.get("markup") == "ALL"), None)
all_d = allrow.get("delta") if allrow else None
gate["delta_ALL"] = all_d
gate["f1_ALL"] = {k: v for k, v in (allrow or {}).items() if k.startswith("f1_")}
layers = {r["markup"]: r.get("delta") for r in rows if r.get("region") == "ALL" and r.get("markup") != "ALL"}
gate["delta_layers"] = layers
gate["delta_regions"] = {r["region"]: r.get("delta") for r in rows if r.get("region") != "ALL" and r.get("markup") == "ALL"}
layer_fails = [(k, v) for k, v in layers.items() if v is not None and v < -0.01]
gate["layer_fails"] = layer_fails
ident = (job / "notes" / "result_identical.txt")
gate["result_identical"] = ident.read_text(encoding="utf-8").strip() if ident.exists() else None
if all_d is None:
    gate["reason"] = "no ALL/ALL row with delta in scores.json"
elif all_d < -0.01:
    gate["reason"] = "delta_ALL=%s < -0.01" % all_d
elif layer_fails:
    gate["reason"] = "layer_drop>0.01: " + ",".join("%s=%s" % (k, v) for k, v in layer_fails)
else:
    gate["pass"] = True
    gate["reason"] = "ok"
(job / "gate.json").write_text(json.dumps(gate, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(gate, ensure_ascii=False))
sys.exit(0 if gate["pass"] else 9)
'@
$gatePyFile = Join-Path $JobDir 'notes\gate_pixgbm.py'
New-Item -ItemType Directory -Force -Path "$JobDir\notes" | Out-Null
Set-Content -Encoding ascii -Path $gatePyFile -Value $gatePy
& $PyHost $gatePyFile $JobDir

$gateEc = $LASTEXITCODE
if ($gateEc -ne 0) { Set-Status "FAILED gate exit=$gateEc"; Get-Content "$JobDir\gate.json" -EA SilentlyContinue; exit 9 }
Set-Status 'DONE'
Get-Content "$JobDir\gate.json"
Get-Content "$JobDir\scores.json" | Select-Object -First 80
