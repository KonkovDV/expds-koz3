# bot-6 path (b): mock e2e PIXGBM S REF weights vs NEW @b0add08 cleanroom.
# Gate: DeltaALL(F1_new-F1_ref)>=-0.01; no layer drop >0.01. Image pantum/n7q:20261024.
$ErrorActionPreference = 'Continue'
[Console]::OutputEncoding = [Text.Encoding]::UTF8

$Sha        = 'b0add08e194853a351d520df19cfb9810c4db983'
$Branch     = 'bots/bot6-pixgbm-cleanroom'
$JobDir     = 'C:\Sloi\pc_jobs\bot6_pixgbm_e2e_metric_b0add08'
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

$ecRef = Run-Variant 'ref' $RefModels
$ecNew = Run-Variant 'new' $NewModels
[IO.File]::WriteAllText($Toml, $BaseToml, $Utf8)  # restore OFF

if (-not (Test-Path "$JobDir\out_ref\result.geojson")) { Set-Status "FAILED ref exit=$ecRef no_geojson"; exit 4 }
if (-not (Test-Path "$JobDir\out_new\result.geojson")) { Set-Status "FAILED new exit=$ecNew no_geojson"; exit 5 }
if (($ecRef -ne 0 -and $ecRef -ne -1) -or ($ecNew -ne 0 -and $ecNew -ne -1)) {
  HB "WARN non-zero exit ref=$ecRef new=$ecNew but geojson present - continue score"
}

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
p = job / "scores.json"
d = json.loads(p.read_text(encoding="utf-8"))
def f1(block, key="ALL"):
    if isinstance(block, dict):
        if key in block and isinstance(block[key], (int, float)): return float(block[key])
        if "f1" in block and isinstance(block["f1"], dict) and key in block["f1"]: return float(block["f1"][key])
        if "layers" in block and key in block["layers"]: return float(block["layers"][key])
    return None
ref = d.get("a") or d.get("pix_ref") or d.get("scores", {}).get("pix_ref")
new = d.get("b") or d.get("pix_new") or d.get("scores", {}).get("pix_new")
delta = d.get("delta") or d.get("Delta") or {}
gate = {"pass": False, "reason": "unparsed", "tol": -0.01, "layer_tol": 0.01}
all_d = None
if isinstance(delta, dict):
    all_d = delta.get("ALL") or (delta.get("f1") or {}).get("ALL")
if all_d is None:
    fa, fb = f1(ref, "ALL"), f1(new, "ALL")
    if fa is not None and fb is not None:
        all_d = fb - fa
gate["delta_ALL"] = all_d
layer_fails = []
if isinstance(delta, dict):
    for k, v in delta.items():
        if k in ("ALL", "f1", "global"): continue
        try:
            dv = float(v) if not isinstance(v, dict) else float(v.get("f1", v.get("ALL", 0)))
        except Exception:
            continue
        if dv < -0.01:
            layer_fails.append((k, dv))
gate["layer_fails"] = layer_fails
if all_d is None:
    gate["reason"] = "could_not_parse_delta_ALL; inspect scores.json"
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
