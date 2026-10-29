# bot8 TTA/MS gate in Docker pantum/n7q:20261024 (PROTOCOL section 10).
# Variants: baseline (flags OFF) / tta / ms / tta_ms on mock regions; optional honest.
# Resumable: STATUS.txt + heartbeat.log + per-variant result.geojson skip.
# Compliance on flags-ON (tta_ms) geojson INSIDE Docker. Host .venv only for scoring.
# JobDir/Wt on C: (F: full). Inputs READ from F: ok. ASCII-only (PS5 ANSI-safe).
# Container name prefix koz3- only.
$ErrorActionPreference = 'Continue'
[Console]::OutputEncoding = [Text.Encoding]::UTF8

$Sha       = '4169266b941e8f5f5b2c743b8692320f91badcb3'
if ($env:KOZ3_SHA) { $Sha = $env:KOZ3_SHA }
$Short     = $Sha.Substring(0,7)
$JobDir    = "C:\Sloi\pc_jobs\bot8_tta_ms_docker_$Short"
$Wt        = 'C:\Sloi\pc_jobs\wt_bot8_orsp_tta'
$Repo      = 'C:\Sloi\wt_bots'
$Branch    = 'bots/bot8-orsp-tta'
$Image     = 'pantum/n7q:20261024'
$Regions   = 'F:\Sloi_pc_jobs_archive\v3e2e\regions'
$HonestReg = $env:KOZ3_HONEST_REGIONS
if (-not $HonestReg) { $HonestReg = 'F:\expds3\honest10_regions' }
$ModelsFill = $env:KOZ3_MODELS
if (-not $ModelsFill) {
  foreach ($c in @(
    'C:\Sloi\wt_bot5_50f6f27\models',
    'C:\Sloi\pc_jobs\_models_fill',
    'F:\Sloi_pc_jobs_archive\v3e2e\kit\models'
  )) { if (Test-Path $c) { $ModelsFill = $c; break } }
}
$ZipV3     = 'C:\Sloi\pc_jobs\_vendor_zips\submission_bot7cand_v3on_0876090.zip'
if (-not (Test-Path $ZipV3)) {
  $ZipV3 = 'F:\Sloi_pc_jobs_archive\v3e2e\kit\submission_bot7cand_v3on_0876090.zip'
}
$PyHost    = 'C:\Sloi\expds-koz3_v0.2.0\expds-koz3\.venv\Scripts\python.exe'
$ToolsTar  = 'F:\Sloi_pc_jobs_archive\v3e2e\kit\tools_v3e2e.tar'
$ToolsSrc  = 'F:\Sloi_pc_jobs_archive\v3e2e\out\tools'
$KitGt     = 'F:\Sloi_pc_jobs_archive\v3e2e\out\gt.geojson'
if (-not (Test-Path $KitGt)) { $KitGt = 'F:\Sloi_pc_jobs_archive\v3e2e\out_v3fix2\gt.geojson' }

New-Item -ItemType Directory -Force -Path $JobDir, "$JobDir\notes", `
  "$JobDir\out_baseline", "$JobDir\out_tta", "$JobDir\out_ms", "$JobDir\out_tta_ms" | Out-Null

function HB($m){ Add-Content -Encoding utf8 "$JobDir\heartbeat.log" "$(Get-Date -Format o) $m"; Write-Host $m }
function Set-Status($s){
  $l = "$s $(Get-Date -Format o)"
  if ($s -match '^(RUNNING|DONE)') {
    $envLine = "ENV=docker:pantum/n7q:20261024"
    Set-Content -Encoding utf8 "$JobDir\STATUS.txt" @($l, $envLine)
  } else {
    Set-Content -Encoding utf8 "$JobDir\STATUS.txt" $l
  }
  HB $l
}

$st = if (Test-Path "$JobDir\STATUS.txt") { Get-Content "$JobDir\STATUS.txt" -Raw } else { '' }
if ($st -match '^DONE') { Write-Host 'already DONE'; exit 0 }

Set-Status 'RUNNING'
HB "start sha=$Sha image=$Image job=$JobDir"

# --- worktree ---
if (-not (Test-Path (Join-Path $Wt '.git'))) {
  New-Item -ItemType Directory -Force -Path (Split-Path $Wt) | Out-Null
  & cmd.exe /c "git -C `"$Repo`" fetch origin $Branch" *> "$JobDir\notes\git_fetch.log"
  & cmd.exe /c "git -C `"$Repo`" worktree add --detach `"$Wt`" $Sha" *> "$JobDir\notes\git_wt.log"
} else {
  & cmd.exe /c "git -C `"$Wt`" fetch origin $Branch" *> "$JobDir\notes\git_fetch.log"
  & cmd.exe /c "git -C `"$Wt`" checkout --force --detach $Sha" *> "$JobDir\notes\git_wt.log"
}
$head = (& cmd.exe /c "git -C `"$Wt`" rev-parse HEAD").Trim()
HB "worktree HEAD=$head"
if ($head -ne $Sha) { Set-Status "FAILED bad_sha want=$Sha got=$head"; exit 2 }

# --- models fill ---
if ($ModelsFill -and (Test-Path $ModelsFill)) {
  Get-ChildItem "$Wt\models" -Recurse -File -EA SilentlyContinue |
    Where-Object { $_.Length -eq 0 -and $_.Extension -match '\.(pt|pth|safetensors|onnx|bin)$' } |
    ForEach-Object { HB "remove_empty $($_.FullName)"; Remove-Item -Force $_.FullName }
  & robocopy $ModelsFill "$Wt\models" /E /IS /IT /NFL /NDL /NJH /NJS *> "$JobDir\notes\robocopy_models.log"
  HB "models filled from $ModelsFill"
} else {
  Set-Status "FAILED no_ModelsFill"
  exit 3
}

# Critical weights for sp_or TTA/MS e2e (must exist, size>0, torch.load in docker)
$Crit = @(
  'models\spor_gor_r18.pt',
  'models\spor_focal_r18.pt',
  'models\surf_arch_r18.pt',
  'models\li_gor_mask_r18.pt',
  'models\li_r18_hard2.pt',
  'models\ae_fort_mask_r18.pt'
)
foreach ($rel in $Crit) {
  $f = Join-Path $Wt $rel
  if (-not (Test-Path $f)) { Set-Status "FAILED missing_asset $f"; exit 3 }
  if ((Get-Item $f).Length -le 0) { Set-Status "FAILED empty_asset $f"; exit 3 }
}

# Preflight: every .pt/.pth/.safetensors under models must be >0 bytes
$wbad = @()
Get-ChildItem "$Wt\models" -Recurse -File -EA SilentlyContinue |
  Where-Object { $_.Extension -match '\.(pt|pth|safetensors)$' } | ForEach-Object {
    if ($_.Length -le 0) { $wbad += $_.FullName }
  }
if ($wbad.Count -gt 0) {
  $wbad | Set-Content -Encoding utf8 "$JobDir\notes\empty_weights.txt"
  Set-Status ("FAILED empty_weights " + ($wbad -join ';'))
  exit 3
}
$wn = @(Get-ChildItem "$Wt\models" -Recurse -File -EA SilentlyContinue |
  Where-Object { $_.Extension -match '\.(pt|pth|safetensors)$' }).Count
HB "preflight_weights_size_ok n=$wn"

# Mandatory torch.load INSIDE Docker (LB env) via repo helper scripts/torch_preflight.py
$PreflightPy = Join-Path $Wt 'scripts\torch_preflight.py'
if (-not (Test-Path $PreflightPy)) {
  Set-Status "FAILED no_torch_preflight_py $PreflightPy"
  exit 3
}
$CNamePf = 'koz3-bot8-torch-preflight-' + (Get-Date -Format 'yyyyMMddHHmmss')
$WtM = ($Wt -replace '\\','/')
$dockerPfArgs = @(
  'run','--rm','--name',$CNamePf,'--network','none',
  '-e','PYTHONDONTWRITEBYTECODE=1','-e','PYTHONUNBUFFERED=1',
  '-v',"${WtM}:/data/zip:ro",
  '-w','/data/zip', $Image,
  'python','scripts/torch_preflight.py'
)
foreach ($rel in $Crit) {
  $dockerPfArgs += ('/data/zip/' + ($rel -replace '\\','/'))
}
HB "docker torch_preflight $CNamePf"
$pPf = Start-Process -FilePath 'docker' -ArgumentList $dockerPfArgs -PassThru -NoNewWindow `
  -RedirectStandardOutput "$JobDir\notes\torch_preflight.log" `
  -RedirectStandardError  "$JobDir\notes\torch_preflight.err"
Wait-Process -InputObject $pPf
HB "torch_preflight exit=$($pPf.ExitCode)"
Get-Content "$JobDir\notes\torch_preflight.log","$JobDir\notes\torch_preflight.err" -EA SilentlyContinue |
  Select-Object -First 40 | ForEach-Object { HB "pf $_" }
if ($pPf.ExitCode -ne 0) {
  Set-Status "FAILED torch_preflight exit=$($pPf.ExitCode)"
  exit 3
}
HB 'preflight_torch_ok'

# --- toml variants ---
$Toml = Join-Path $Wt 'configs\default.toml'
if (-not (Test-Path $Toml)) { Set-Status 'FAILED no_default_toml'; exit 6 }
$BaseToml = [IO.File]::ReadAllText($Toml)
$Utf8 = [Text.UTF8Encoding]::new($false)

function Get-BaselineToml([string]$t) {
  $t = [regex]::Replace($t, "(?ms)\r?\n\[tta\]\r?\n(?:(?!\[).)*", "")
  $t = [regex]::Replace($t, "(?ms)\r?\n\[ms\]\r?\n(?:(?!\[).)*", "")
  return $t.TrimEnd() + "`n"
}
$OffToml = Get-BaselineToml $BaseToml

$TtaBlock = @"

[tta]
sp_or = 4
"@
$MsBlock = @"

[ms]
sp_or = [0.75, 1.0, 1.25]
"@

$VariantTomls = @{
  baseline = $OffToml
  tta      = ($OffToml + $TtaBlock + "`n")
  ms       = ($OffToml + $MsBlock + "`n")
  tta_ms   = ($OffToml + $TtaBlock + $MsBlock + "`n")
}

function Run-Variant([string]$name, [string]$tomlText, [string]$regionsPath, [string]$outSub) {
  $outDir = Join-Path $JobDir $outSub
  New-Item -ItemType Directory -Force -Path $outDir | Out-Null
  $res = Join-Path $outDir 'result.geojson'
  if (Test-Path $res) { HB "skip $outSub (result exists)"; return 0 }
  if (-not (Test-Path $regionsPath)) { HB "skip $outSub (no regions $regionsPath)"; return -2 }
  [IO.File]::WriteAllText($Toml, $tomlText, $Utf8)
  Copy-Item -Force $Toml (Join-Path $outDir 'default.toml')
  $CName = "koz3-bot8-ttams-$name-" + (Get-Date -Format 'yyyyMMddHHmmss')
  $WtM2 = ($Wt -replace '\\','/')
  $RegM = ($regionsPath -replace '\\','/')
  $OutM = ($JobDir -replace '\\','/')
  $args2 = @(
    'run','--rm','--name',$CName,'--gpus','all','--network','none',
    '-e','PYTHONDONTWRITEBYTECODE=1','-e','PYTHONUNBUFFERED=1',
    '-e','HF_HUB_OFFLINE=1','-e','TRANSFORMERS_OFFLINE=1',
    '-v',"${WtM2}:/data/zip:ro",
    '-v',"${RegM}:/data/regions:ro",
    '-v',"${OutM}:/data/output",
    '-w','/data/zip', $Image,
    'python','inference/main.py','/data/regions',"/data/output/$outSub/result.geojson"
  )
  HB "docker run $outSub $CName"
  $t1 = Get-Date
  $p = Start-Process -FilePath 'docker' -ArgumentList $args2 -PassThru -NoNewWindow `
    -RedirectStandardOutput (Join-Path $outDir 'main.log') `
    -RedirectStandardError  (Join-Path $outDir 'main.err')
  while (-not $p.HasExited) {
    HB "$outSub running $([int]((Get-Date)-$t1).TotalSeconds)s"
    Start-Sleep 20
  }
  $sec = [math]::Round(((Get-Date)-$t1).TotalSeconds,1)
  Set-Content (Join-Path $outDir 'wall_sec.txt') $sec
  HB "$outSub exit=$($p.ExitCode) wall_sec=$sec"
  return $p.ExitCode
}

# --- mock variants ---
$ec = @{}
foreach ($v in @('baseline','tta','ms','tta_ms')) {
  $ec[$v] = Run-Variant $v $VariantTomls[$v] $Regions "out_$v"
}
[IO.File]::WriteAllText($Toml, $OffToml, $Utf8)

# --- optional honest-10 ---
if (Test-Path $HonestReg) {
  New-Item -ItemType Directory -Force -Path "$JobDir\out_honest10" | Out-Null
  foreach ($v in @('baseline','tta','ms','tta_ms')) {
    $ec["h_$v"] = Run-Variant "h_$v" $VariantTomls[$v] $HonestReg "out_honest10\$v"
  }
  [IO.File]::WriteAllText($Toml, $OffToml, $Utf8)
  HB 'honest-10 variants done'
} else {
  HB "honest-10 regions missing ($HonestReg) - mock-only; set KOZ3_HONEST_REGIONS to enable"
  "skipped no regions at $HonestReg" | Set-Content -Encoding utf8 "$JobDir\out_honest10_skipped.txt"
}

# --- compliance INSIDE Docker on flags-ON (tta_ms) ---
$OnGeo = Join-Path $JobDir 'out_tta_ms\result.geojson'
if (Test-Path $OnGeo) {
  $CName = 'koz3-bot8-compliance-' + (Get-Date -Format 'yyyyMMddHHmmss')
  $WtM3 = ($Wt -replace '\\','/')
  $OutM3 = ($JobDir -replace '\\','/')
  $ccArgs = @(
    'run','--rm','--name',$CName,'--network','none',
    '-e','PYTHONDONTWRITEBYTECODE=1','-e','PYTHONUNBUFFERED=1','-e','PYTHONIOENCODING=utf-8',
    '-v',"${WtM3}:/data/zip:ro",
    '-v',"${OutM3}:/data/output:ro",
    '-w','/data/zip', $Image,
    'python','scripts/check_compliance.py','/data/output/out_tta_ms/result.geojson'
  )
  HB "docker compliance $CName"
  $p = Start-Process -FilePath 'docker' -ArgumentList $ccArgs -PassThru -NoNewWindow `
    -RedirectStandardOutput "$JobDir\compliance_tta_ms.log" `
    -RedirectStandardError  "$JobDir\compliance_tta_ms.err"
  Wait-Process -InputObject $p
  HB "compliance exit=$($p.ExitCode)"
  Set-Content "$JobDir\compliance_exit.txt" "$($p.ExitCode)"
} else {
  HB 'compliance skipped - no out_tta_ms/result.geojson'
}

# --- score mock variants ---
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
HB "scorer=$ScorePy metric=$MetricPy gt=$KitGt"

if ($ScorePy -and $MetricPy -and (Test-Path $KitGt) -and (Test-Path "$JobDir\out_baseline\result.geojson")) {
  $scoreArgs = @(
    $ScorePy, '--metric', $MetricPy, '--gt', $KitGt,
    '--a', "baseline=$JobDir\out_baseline\result.geojson"
  )
  foreach ($v in @('tta','ms','tta_ms')) {
    $g = "$JobDir\out_$v\result.geojson"
    if (Test-Path $g) { $scoreArgs += @('--b', "$v=$g") }
  }
  $scoreArgs += @('--out', "$JobDir\scores.json")
  & $PyHost @scoreArgs *> "$JobDir\scores.log"
  if (-not (Test-Path "$JobDir\scores.json")) {
    HB 'WARN score failed'
    Get-Content "$JobDir\scores.log" -EA SilentlyContinue | Select-Object -First 40
  } else {
    HB 'scores.json written'
    Get-Content "$JobDir\scores.json" | Select-Object -First 40
  }
} else {
  HB 'score skipped (no scorer/gt/baseline)'
}

$need = @('out_baseline','out_tta','out_ms','out_tta_ms')
$missing = @($need | Where-Object { -not (Test-Path "$JobDir\$_\result.geojson") })
if ($missing.Count -gt 0) {
  Set-Status ("FAILED missing " + ($missing -join ','))
  exit 5
}
Set-Status 'DONE'
HB 'all done'
exit 0
