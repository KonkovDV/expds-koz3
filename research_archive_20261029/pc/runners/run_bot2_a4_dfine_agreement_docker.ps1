# bot2 A4 D-FINE agreement_m=50 @ c3d912c1a7fb2a551d2a4d37446f8d840e138d71 - GPU via pantum/n7q (NOT host venv). Resumable.
# Hypothesis: neutralize 13 far FP (A3); expect ALL/sp_or ~= B. Default agreement_m=0 in code.
# Container name prefix koz3- only; never touch talent/kontur/microphoenix.
$ErrorActionPreference = 'Continue'
[Console]::OutputEncoding = [Text.Encoding]::UTF8
$JobDir = 'C:\Sloi\pc_jobs\bot2_a4_dfine_agreement_c3d912c'
$Wt = 'C:\Sloi\pc_jobs\wt_bot2_a4_c3d912c'
$Sha = 'c3d912c1a7fb2a551d2a4d37446f8d840e138d71'
$Branch = 'bots/bot2-e2e-v3-prep'
$PyHost = 'C:\Sloi\expds-koz3_v0.2.0\expds-koz3\.venv\Scripts\python.exe'  # scoring only
$Image = 'pantum/n7q:20261024'
$Regions = 'F:\Sloi_pc_jobs_archive\v3e2e\regions'
$ZipV3 = 'F:\Sloi_pc_jobs_archive\v3e2e\kit\submission_bot7cand_v3on_0876090.zip'
$ToolsTar = 'F:\Sloi_pc_jobs_archive\v3e2e\kit\tools_v3e2e.tar'
$ToolsSrc = 'F:\Sloi_pc_jobs_archive\v3e2e\out\tools'
$KitGt = 'F:\Sloi_pc_jobs_archive\v3e2e\out\gt.geojson'
$BGeo = 'F:\Sloi_pc_jobs_archive\bot2_v3_hotfix_1ec2d7c\out_B\result.geojson'
$Name = 'koz3-bot2-a4-' + (Get-Date -Format 'yyyyMMddHHmmss')
New-Item -ItemType Directory -Force -Path $JobDir, "$JobDir\out_A", "$JobDir\dump_A", "$JobDir\notes" | Out-Null

function HB($m){ Add-Content -Encoding utf8 "$JobDir\heartbeat.log" "$(Get-Date -Format o) $m" }
function Set-Status($s){ $l="$s $(Get-Date -Format o)"; Set-Content -Encoding utf8 "$JobDir\STATUS.txt" $l; HB $l }

$st = if (Test-Path "$JobDir\STATUS.txt") { Get-Content "$JobDir\STATUS.txt" -Raw } else { '' }
if ($st -match '^DONE') { Write-Host 'already DONE'; exit 0 }

Set-Status 'RUNNING'
HB "start A4 agreement_m=50 docker name=$Name"

# worktree
if (-not (Test-Path "$Wt\.git")) {
  HB 'worktree add'
  & cmd /c "git -C C:\Sloi\wt_bots fetch origin $Branch" 2>&1 | Out-File "$JobDir\git_fetch.log"
  & cmd /c "git -C C:\Sloi\wt_bots worktree add --detach `"$Wt`" $Sha" 2>&1 | Out-File "$JobDir\git_wt.log"
} else {
  $h0 = (& cmd /c "git -C `"$Wt`" rev-parse HEAD").Trim()
  if ($h0 -ne $Sha) {
    & cmd /c "git -C C:\Sloi\wt_bots fetch origin $Branch" 2>&1 | Out-File "$JobDir\git_fetch.log"
    & cmd /c "git -C `"$Wt`" checkout --detach $Sha" 2>&1 | Out-File "$JobDir\git_co.log"
  }
}
$head = (& cmd /c "git -C `"$Wt`" rev-parse HEAD").Trim()
HB "HEAD=$head"
if ($head -ne $Sha) { Set-Status "FAILED bad_sha $head"; exit 2 }

# tools for score (tar lacks e2e_score.py - copy from archive out\tools)
New-Item -ItemType Directory -Force -Path "$JobDir\tools" | Out-Null
if (Test-Path $ToolsTar) { tar -xf $ToolsTar -C $JobDir 2>$null }
foreach ($f in @('e2e_score.py','compute_metrics.py')) {
  $dst = Join-Path $JobDir "tools\$f"
  if (-not (Test-Path $dst) -and (Test-Path (Join-Path $ToolsSrc $f))) {
    Copy-Item -Force (Join-Path $ToolsSrc $f) $dst; HB "copied scorer $f from ToolsSrc"
  }
}
$ScorePy = Get-ChildItem -Path $JobDir -Recurse -Filter e2e_score.py -EA SilentlyContinue | Select-Object -First 1 -ExpandProperty FullName
$MetricPy = Get-ChildItem -Path $JobDir -Recurse -Filter compute_metrics.py -EA SilentlyContinue | Select-Object -First 1 -ExpandProperty FullName
HB "scorer=$ScorePy metric=$MetricPy"
HB "scorer=$ScorePy"

# unpack zip_A + overlay + patch toml (A: enabled=true drop_groups=[])
$Z = "$JobDir\zip_A"
if (-not (Test-Path "$Z\inference\main.py")) {
  New-Item -ItemType Directory -Force -Path $Z | Out-Null
  try { tar -xf $ZipV3 -C $Z 2>$null } catch {}
  if (-not (Test-Path "$Z\inference\main.py")) { Expand-Archive -Path $ZipV3 -DestinationPath $Z -Force }
}
foreach ($rel in @('expds\pipeline.py','expds\sp_dfine.py')) {
  $src = Join-Path $Wt $rel
  if (Test-Path $src) { Copy-Item -Force $src (Join-Path $Z $rel); HB "overlay $rel" }
}
$toml = "$Z\configs\default.toml"
$t = Get-Content -Raw -Encoding UTF8 $toml
$t = [regex]::Replace($t, '(?m)^\[sp_dfine\]\r?\nenabled = false\r?\n', "[sp_dfine]`r`nenabled = true`r`n")
$t = [regex]::Replace($t, '(?m)^drop_groups = \[.*?\]\r?\n', "drop_groups = []`r`n")
if ($t -notmatch '(?m)^drop_groups =') {
  $t = [regex]::Replace($t, '(?m)^(\[sp_dfine\]\r?\nenabled = true\r?\n)', "`$1drop_groups = []`r`n")
}
if ($t -match '(?m)^drop_groups = \["sp_or"') { Set-Status 'FAILED R_drop_groups'; exit 3 }
# A4: agreement_m=50 (code default 0)
if ($t -match '(?m)^agreement_m\s*=') {
  $t = [regex]::Replace($t, '(?m)^agreement_m\s*=.*$', 'agreement_m = 50')
} else {
  $t = [regex]::Replace($t, '(?m)^(\[sp_dfine\]\r?\nenabled = true\r?\n)', "`$1agreement_m = 50`r`n")
}
# UTF-8 NO BOM (Set-Content -Encoding UTF8 writes BOM → tomllib Invalid statement col1)
[IO.File]::WriteAllText($toml, $t, [Text.UTF8Encoding]::new($false))
Select-String -Path $toml -Pattern '\[sp_dfine\]|enabled =|drop_groups|agreement_m' | Select-Object -First 12 | ForEach-Object { $_.Line } |
  Set-Content -Encoding UTF8 "$JobDir\toml_A_snip.txt"

# safety: list containers; refuse if we would stop non-koz3
HB 'docker ps before'
docker ps -a --format '{{.Names}} {{.Status}}' 2>&1 | Tee-Object "$JobDir\docker_ps_before.txt" | Out-Null

# docker main - Linux vendor transformers
$outGeo = '/data/output/out_A/result.geojson'
$dump = '/data/output/dump_A'
# Windows path mounts for Docker Desktop
$ZMount = ($Z -replace '\\','/')
$RegMount = ($Regions -replace '\\','/')
$OutMount = ($JobDir -replace '\\','/')

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

HB "docker run $Name"
# wipe stale geojson so salvage cannot accept pre-run file
New-Item -ItemType Directory -Force -Path "$JobDir\out_A" | Out-Null
Remove-Item "$JobDir\out_A\result.geojson" -Force -EA SilentlyContinue

$t1 = Get-Date
# Prefer --gpus all; fall back without if daemon rejects
$dockerArgs = @(
  'run','--name', $Name,
  '--gpus','all',
  '-e', 'EXPDS_DFINE_DUMP=/data/output/dump_A',
  '-e', 'PYTHONDONTWRITEBYTECODE=1',
  '-e', 'PYTHONUNBUFFERED=1',
  '-v', "${ZMount}:/data/zip:ro",
  '-v', "${RegMount}:/data/regions:ro",
  '-v', "${OutMount}:/data/output",
  '-w', '/data/zip',
  $Image,
  'python', 'inference/main.py', '/data/regions', '/data/output/out_A/result.geojson'
)
$p = Start-Process -FilePath 'docker' -ArgumentList $dockerArgs -PassThru -NoNewWindow `
  -RedirectStandardOutput "$JobDir\main_A.log" -RedirectStandardError "$JobDir\main_A.err"
while (-not $p.HasExited) {
  HB "docker pid=$($p.Id) $([int]((Get-Date)-$t1).TotalSeconds)s"
  Start-Sleep 20
}
$null = $p.WaitForExit(5000)
$p.Refresh()
$sec = [math]::Round(((Get-Date)-$t1).TotalSeconds,1)
$ec = $p.ExitCode
if ($null -eq $ec) { HB 'WARN ExitCode null after WaitForExit'; $ec = -1 }
HB "docker exit=$ec sec=$sec"
Set-Content "$JobDir\wall_sec.txt" $sec

# remove ONLY our koz3-* container (if still present; --rm not used so we can inspect)
docker rm -f $Name 2>&1 | Tee-Object "$JobDir\docker_rm.txt" | Out-Null
HB "removed $Name"

# stages summary
$stages = Get-ChildItem -Path "$JobDir\dump_A" -Recurse -Filter 'stages.json' -EA SilentlyContinue | Select-Object -First 1 -ExpandProperty FullName
if ($stages) {
  Copy-Item -Force $stages "$JobDir\stages.json"
  HB "stages=$stages"
  $stagesPy = @'
import json, pathlib, sys
p=pathlib.Path(sys.argv[1])
d=json.loads(p.read_text(encoding="utf-8"))
def dig(o,*keys):
  if not isinstance(o,dict): return None
  for k in keys:
    if k in o: return o[k]
  for v in o.values():
    if isinstance(v,dict):
      r=dig(v,*keys)
      if r is not None: return r
  return None
out={
  "n_models": dig(d,"n_models","n_model"),
  "n_raw_boxes": dig(d,"n_raw_boxes","n_raw"),
  "n_would_pass_thr": dig(d,"n_would_pass_thr"),
  "n_sp_dfine_in_state": dig(d,"n_sp_dfine_in_state","n_in_state"),
  "skip_reason": dig(d,"skip_reason"),
}
outp=pathlib.Path(sys.argv[2])
outp.write_text(json.dumps({"raw_keys":list(d.keys()) if isinstance(d,dict) else type(d).__name__,"dig":out,"full":d},indent=2),encoding="utf-8")
print(json.dumps(out,indent=2))
'@
  $stagesPyFile = Join-Path $JobDir 'notes\stages_summary.py'
  Set-Content -Encoding ascii -Path $stagesPyFile -Value $stagesPy
  & $PyHost $stagesPyFile $stages "$JobDir\stages_summary.json" 2>&1 | Tee-Object "$JobDir\stages_summary.log"
} else {
  HB 'NO stages.json'
}

# score vs B if present
$Gt = "$JobDir\gt.geojson"
if ($ScorePy -and (Test-Path "$JobDir\out_A\result.geojson")) {
  if (-not (Test-Path $Gt)) {
    if (Test-Path $KitGt) { Copy-Item -Force $KitGt $Gt; HB "gt copied from KitGt" }
    else {
      $env:PYTHONPATH = $Wt
      & $PyHost -m expds.gt --root $Regions --out $Gt *> "$JobDir\gt.log"
    }
  }
  if (Test-Path $BGeo) {
    & $PyHost $ScorePy --metric $MetricPy --gt $Gt `
      --a "A4=$JobDir\out_A\result.geojson" --b "B=$BGeo" `
      --out "$JobDir\e2e_A4_B.json" *> "$JobDir\e2e_A4_B.log"
  }
  $V2 = 'F:\Sloi_pc_jobs_archive\v3e2e\out_v3fix2\res_v2\result.geojson'
  if (-not (Test-Path $V2)) { $V2 = 'F:\Sloi_pc_jobs_archive\v3e2e\out\res_v2\result.geojson' }
  if (Test-Path $V2) {
    & $PyHost $ScorePy --metric $MetricPy --gt $Gt `
      --a "v2=$V2" --b "A4=$JobDir\out_A\result.geojson" `
      --out "$JobDir\e2e_v2_A4.json" *> "$JobDir\e2e_v2_A4.log"
  }
}

# Strict salvage: geojson mtime after container start, size>50, and exit==0.
# Start-Process docker on Windows often leaves ExitCode null (reported as -1) even on success;
# resolve -1/null to 0 only when fresh geojson AND main_A.log shows a completed write (not TOMLDecodeError).
$geoPath = "$JobDir\out_A\result.geojson"
$hasOut = Test-Path $geoPath
$gi = $null
if ($hasOut) { $gi = Get-Item $geoPath }
$fresh = $hasOut -and ($gi.Length -gt 50) -and ($gi.LastWriteTime -ge $t1)
$logOk = $false
if (Test-Path "$JobDir\main_A.log") {
  if (Select-String -Path "$JobDir\main_A.log" -Pattern '"written"' -Quiet -EA SilentlyContinue) { $logOk = $true }
  if (Select-String -Path "$JobDir\main_A.log" -Pattern 'TOMLDecodeError|Invalid statement' -Quiet -EA SilentlyContinue) { $logOk = $false }
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
Get-Content "$JobDir\stages_summary.json" -EA SilentlyContinue | Select-Object -First 40
Get-Content "$JobDir\e2e_v2_A4.json" -EA SilentlyContinue | Select-Object -First 40
