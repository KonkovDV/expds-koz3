# bot8 TTA-on-bot3 gate: bot3 dfine_aeor Ae f4pack candidate + TTA/MS overlays.
# Image pantum/n7q:20261024. JobDir/Wt/weights on C:\Sloi\pc_jobs. ASCII-only.
# No python -c. Weights preflight incl. f4pack (sha 249b86fb). schtasks launch.
# Variants: base_ae (dfine Ae ON, new TTA flags OFF) / tta_ae / tta_ms_prio.
# Mock + optional honest-10; wall_summary; compliance on submission output.
# Run AFTER bot8 first gate (bots/bot8-tta-lb @ f6fe20a).
$ErrorActionPreference = 'Continue'
[Console]::OutputEncoding = [Text.Encoding]::UTF8

$Sha = 'ef2bd8882d7a43609dca429a8fb319bdfc59aca6'
if ($env:KOZ3_SHA) { $Sha = $env:KOZ3_SHA }
$Short = $Sha.Substring(0,7)
$JobDir = "C:\Sloi\pc_jobs\bot8_tta_on_bot3_$Short"
$Wt = 'C:\Sloi\pc_jobs\wt_bot8_tta_on_bot3'
$Repo = 'C:\Sloi\wt_bots'
$Branch = 'bots/bot8-tta-on-bot3'
$Image = 'pantum/n7q:20261024'
$EnvTag = 'ENV=docker:pantum/n7q:20261024'
$Regions = 'F:\Sloi_pc_jobs_archive\v3e2e\regions'
$HonestReg = $env:KOZ3_HONEST_REGIONS
if (-not $HonestReg) { $HonestReg = 'F:\expds3\honest10_regions' }
$WeightsAe = if ($env:KOZ3_WEIGHTS_AE) { $env:KOZ3_WEIGHTS_AE } else { 'C:\Sloi\pc_jobs\weights\dfine_ae_ft_f4' }
$ShaPrefix = if ($env:KOZ3_AE_SHA_PREFIX) { $env:KOZ3_AE_SHA_PREFIX } else { '249b86fb' }
$ModelsFill = $env:KOZ3_MODELS
if (-not $ModelsFill) {
  foreach ($c in @(
    'C:\Sloi\wt_bot5_50f6f27\models',
    'C:\Sloi\pc_jobs\_models_fill',
    'F:\Sloi_pc_jobs_archive\v3e2e\kit\models'
  )) { if (Test-Path $c) { $ModelsFill = $c; break } }
}
$ZipV3 = 'C:\Sloi\pc_jobs\_vendor_zips\submission_bot7cand_v3on_0876090.zip'
if (-not (Test-Path $ZipV3)) {
  $ZipV3 = 'F:\Sloi_pc_jobs_archive\v3e2e\kit\submission_bot7cand_v3on_0876090.zip'
}
$PyHost = 'C:\Sloi\expds-koz3_v0.2.0\expds-koz3\.venv\Scripts\python.exe'
$ToolsTar = 'F:\Sloi_pc_jobs_archive\v3e2e\kit\tools_v3e2e.tar'
$ToolsSrc = 'F:\Sloi_pc_jobs_archive\v3e2e\out\tools'
$KitGt = 'F:\Sloi_pc_jobs_archive\v3e2e\out\gt.geojson'
if (-not (Test-Path $KitGt)) { $KitGt = 'F:\Sloi_pc_jobs_archive\v3e2e\out_v3fix2\gt.geojson' }

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$PreflightAe = Join-Path $ScriptDir 'bot3_ae_weights_preflight.py'
if (-not (Test-Path $PreflightAe)) {
  $PreflightAe = 'C:\Sloi\pc_jobs\runners\bot3_ae_weights_preflight.py'
}

New-Item -ItemType Directory -Force -Path $JobDir, "$JobDir\notes", `
  "$JobDir\out_base_ae", "$JobDir\out_tta_ae", "$JobDir\out_tta_ms_prio" | Out-Null

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
HB "start sha=$Sha branch=$Branch image=$Image job=$JobDir f4pack=$ShaPrefix"

# --- worktree ---
& cmd.exe /c "git -C `"$Repo`" fetch origin $Branch" *> "$JobDir\notes\git_fetch.log"
if ($Sha -eq 'PLACEHOLDER_SHA' -or -not $Sha) {
  $Sha = (& cmd.exe /c "git -C `"$Repo`" rev-parse origin/$Branch").Trim()
  if (-not $Sha) { Set-Status "FAILED resolve_sha origin/$Branch"; exit 2 }
  $Short = $Sha.Substring(0,7)
  HB "resolved Sha from origin/$Branch -> $Sha"
}
if (-not (Test-Path "$Wt\.git")) {
  New-Item -ItemType Directory -Force -Path (Split-Path $Wt) | Out-Null
  & cmd.exe /c "git -C `"$Repo`" worktree add --detach `"$Wt`" $Sha" *> "$JobDir\notes\git_wt.log"
} else {
  & cmd.exe /c "git -C `"$Wt`" fetch origin $Branch" *> "$JobDir\notes\git_fetch2.log"
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

# --- f4pack Ae weights ---
New-Item -ItemType Directory -Force -Path "$Wt\models\dfine_ae", "$Wt\models\dfine_aeor" | Out-Null
$AeSrcSt = Join-Path $WeightsAe 'model.safetensors'
if (-not (Test-Path $AeSrcSt)) {
  $alt = Join-Path $WeightsAe 'dfine_ae\model.safetensors'
  if (Test-Path $alt) { $WeightsAe = Join-Path $WeightsAe 'dfine_ae'; $AeSrcSt = Join-Path $WeightsAe 'model.safetensors' }
}
if (-not (Test-Path $AeSrcSt)) {
  Set-Status "FAILED missing_ae_weights $WeightsAe"
  exit 3
}
if ((Get-Item $AeSrcSt).Length -le 0) {
  Set-Status "FAILED empty_ae_weights $AeSrcSt"
  exit 3
}
& robocopy $WeightsAe "$Wt\models\dfine_ae" config.json model.safetensors /NFL /NDL /NJH /NJS *> "$JobDir\notes\robocopy_dfine_ae.log"
$ThrCandidates = @(
  (Join-Path $WeightsAe 'thresholds_oof.json'),
  (Join-Path $WeightsAe 'models\dfine_aeor\thresholds_oof.json'),
  (Join-Path (Split-Path $WeightsAe -Parent) 'thresholds_oof.json')
)
$ThrSrc = $null
foreach ($t in $ThrCandidates) { if (Test-Path $t) { $ThrSrc = $t; break } }
if (-not $ThrSrc) { Set-Status "FAILED missing_thresholds near $WeightsAe"; exit 3 }
Copy-Item -Force $ThrSrc "$Wt\models\dfine_aeor\thresholds_oof.json"
HB "thresholds from $ThrSrc"

# --- HOST preflight f4pack (helper .py, no python -c) ---
if (-not (Test-Path $PyHost)) { Set-Status "FAILED no_PyHost $PyHost"; exit 3 }
if (-not (Test-Path $PreflightAe)) { Set-Status "FAILED no_preflight_ae $PreflightAe"; exit 3 }
Copy-Item -Force $PreflightAe "$JobDir\notes\bot3_ae_weights_preflight.py"
$hostPf = & $PyHost "$JobDir\notes\bot3_ae_weights_preflight.py" --dir "$Wt\models\dfine_ae" --sha-prefix $ShaPrefix --require-config 2>&1
$hostPf | Set-Content -Encoding ascii "$JobDir\notes\host_preflight.log"
HB "host_preflight $hostPf"
if ($LASTEXITCODE -ne 0) { Set-Status "FAILED host_preflight exit=$LASTEXITCODE"; exit 3 }

# --- DOCKER preflight f4pack full-load ---
$PfOut = "$JobDir\notes\docker_preflight"
New-Item -ItemType Directory -Force -Path $PfOut | Out-Null
$CNamePf = 'koz3-bot8-f4-preflight-' + (Get-Date -Format 'yyyyMMddHHmmss')
$wtM = ($Wt -replace '\\','/')
$jdM = ($JobDir -replace '\\','/')
$pfArgs = @(
  'run','--rm','--name',$CNamePf,'--network','none',
  '-e','PYTHONDONTWRITEBYTECODE=1','-e','PYTHONUNBUFFERED=1',
  '-v',"${wtM}/models/dfine_ae:/weights:ro",
  '-v',"${jdM}/notes/bot3_ae_weights_preflight.py:/preflight.py:ro",
  '-v',"${jdM}/notes/docker_preflight:/out",
  $Image,
  'python','/preflight.py','--dir','/weights','--sha-prefix',$ShaPrefix,'--require-config','--full-load'
)
HB "docker f4pack preflight $CNamePf"
$pp = Start-Process -FilePath 'docker' -ArgumentList $pfArgs -PassThru -NoNewWindow `
  -RedirectStandardOutput "$PfOut\stdout.log" -RedirectStandardError "$PfOut\stderr.log"
while (-not $pp.HasExited) { Start-Sleep 5 }
HB "docker_preflight exit=$($pp.ExitCode)"
Get-Content "$PfOut\stdout.log","$PfOut\stderr.log" -EA SilentlyContinue |
  Set-Content -Encoding ascii "$JobDir\notes\docker_preflight.log"
if ($pp.ExitCode -ne 0) { Set-Status "FAILED docker_preflight exit=$($pp.ExitCode)"; exit 3 }
HB 'docker_preflight PASS'

# --- vendor ---
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

# empty weight guard
$wbad = @()
Get-ChildItem "$Wt\models" -Recurse -File -EA SilentlyContinue |
  Where-Object { $_.Extension -match '\.(pt|pth|safetensors)$' -and $_.Length -le 0 } |
  ForEach-Object { $wbad += $_.FullName }
if ($wbad.Count -gt 0) {
  $wbad | Set-Content -Encoding ascii "$JobDir\notes\empty_weights.txt"
  Set-Status ("FAILED empty_weights " + ($wbad -join ';'))
  exit 3
}

# --- toml variants ---
# base_ae: bot3 candidate Ae ON, NEW TTA flags OFF (rot90_tta=false, scales=[1.0], no [tta]/[ms])
# tta_ae: + U-Net tta ae=4 + dfine rot90_tta=true (flip stays bot3 default)
# tta_ms_prio: + tta ae/or=4 + ms ae/or + dfine rot90 + scales MS; budget_s raised for plan_level
$Toml = "$Wt\configs\default.toml"
$OffToml = [IO.File]::ReadAllText($Toml)
if ($OffToml -notmatch "\[dfine_aeor\]\r?\nenabled = false") {
  Set-Status 'FAILED toml_anchor_missing'
  exit 6
}
$BaseAeBlock = @(
  '[dfine_aeor]',
  'enabled = true',
  'markups = ["ae"]',
  'weights = { ae = ["models/dfine_ae"] }',
  'thresholds_file = "models/dfine_aeor/thresholds_oof.json"',
  'rot90_tta = false',
  'scales = [1.0]',
  'budget_s = 180.0',
  ''
) -join "`n"
$TtaAeBlock = @(
  '[dfine_aeor]',
  'enabled = true',
  'markups = ["ae"]',
  'weights = { ae = ["models/dfine_ae"] }',
  'thresholds_file = "models/dfine_aeor/thresholds_oof.json"',
  'rot90_tta = true',
  'scales = [1.0]',
  'budget_s = 420.0',
  '',
  '[tta]',
  'ae = 4',
  ''
) -join "`n"
$TtaMsBlock = @(
  '[dfine_aeor]',
  'enabled = true',
  'markups = ["ae"]',
  'weights = { ae = ["models/dfine_ae"] }',
  'thresholds_file = "models/dfine_aeor/thresholds_oof.json"',
  'rot90_tta = true',
  'scales = [0.75, 1.0, 1.25]',
  'budget_s = 600.0',
  '',
  '[tta]',
  'ae = 4',
  'or = 4',
  '',
  '[ms]',
  'ae = [0.75, 1.0, 1.25]',
  'or = [0.75, 1.0, 1.25]',
  ''
) -join "`n"

function Inject-Dfine([string]$src, [string]$block) {
  return [regex]::Replace($src, "\[dfine_aeor\]\r?\nenabled = false", $block)
}
$Utf8 = [Text.UTF8Encoding]::new($false)
$VariantTomls = @{
  base_ae      = (Inject-Dfine $OffToml $BaseAeBlock)
  tta_ae       = (Inject-Dfine $OffToml $TtaAeBlock)
  tta_ms_prio  = (Inject-Dfine $OffToml $TtaMsBlock)
}
$VariantOrder = @('base_ae','tta_ae','tta_ms_prio')

function Run-Variant([string]$name, [string]$tomlText, [string]$regionsPath, [string]$outSub) {
  $outDir = Join-Path $JobDir $outSub
  New-Item -ItemType Directory -Force -Path $outDir | Out-Null
  $res = Join-Path $outDir 'result.geojson'
  if (Test-Path $res) { HB "skip $outSub (result exists)"; return 0 }
  if (-not (Test-Path $regionsPath)) { HB "skip $outSub (no regions $regionsPath)"; return -2 }
  [IO.File]::WriteAllText($Toml, $tomlText, $Utf8)
  Copy-Item -Force $Toml (Join-Path $outDir 'default.toml')
  $CName = "koz3-bot8-ttabot3-$name-" + (Get-Date -Format 'yyyyMMddHHmmss')
  $WtM2 = ($Wt -replace '\\','/')
  $RegM = ($regionsPath -replace '\\','/')
  $OutM = ($JobDir -replace '\\','/')
  $args2 = @(
    'run','--rm','--name',$CName,'--gpus','all','--network','none',
    '-e','PYTHONDONTWRITEBYTECODE=1','-e','PYTHONUNBUFFERED=1',
    '-e','HF_HUB_OFFLINE=1','-e','TRANSFORMERS_OFFLINE=1','-e','EXPDS_STRICT=1',
    '-e',"EXPDS_STATUS_DIR=/data/output/$outSub",
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
  Set-Content -Encoding ascii (Join-Path $outDir 'wall_sec.txt') "$sec"
  HB "$outSub exit=$($p.ExitCode) wall_sec=$sec"
  return $p.ExitCode
}

# --- mock ---
if (-not (Test-Path $Regions)) { Set-Status "FAILED no_mock_regions $Regions"; exit 3 }
$ec = @{}
foreach ($v in $VariantOrder) {
  $ec[$v] = Run-Variant $v $VariantTomls[$v] $Regions "out_$v"
}
[IO.File]::WriteAllText($Toml, $OffToml, $Utf8)

$wallLines = @("variant,wall_sec,exit")
foreach ($v in $VariantOrder) {
  $ws = Join-Path $JobDir "out_$v\wall_sec.txt"
  $sec = if (Test-Path $ws) { (Get-Content $ws -Raw).Trim() } else { 'NA' }
  $wallLines += "$v,$sec,$($ec[$v])"
  HB "wall $v sec=$sec exit=$($ec[$v])"
}
$wallLines | Set-Content -Encoding ascii "$JobDir\wall_summary.csv"

# assert Ae nonzero on base_ae
$AssertPy = "$Wt\scripts\assert_dfine_aeor_ae_nonzero.py"
if (Test-Path $AssertPy -and (Test-Path "$JobDir\out_base_ae\result.geojson")) {
  & $PyHost $AssertPy --geojson "$JobDir\out_base_ae\result.geojson" --main-log "$JobDir\out_base_ae\main.log" *> "$JobDir\assert_ae_nonzero.log"
  if ($LASTEXITCODE -ne 0) { Set-Status "FAILED assert_ae_nonzero exit=$LASTEXITCODE"; exit 9 }
  HB 'assert Ae nonzero PASS'
}

# --- honest-10 ---
if ((Test-Path $HonestReg) -and (Get-ChildItem $HonestReg -EA SilentlyContinue | Measure-Object).Count -gt 0) {
  New-Item -ItemType Directory -Force -Path "$JobDir\out_honest10" | Out-Null
  foreach ($v in $VariantOrder) {
    $ec["h_$v"] = Run-Variant "h_$v" $VariantTomls[$v] $HonestReg "out_honest10\$v"
  }
  [IO.File]::WriteAllText($Toml, $OffToml, $Utf8)
  HB 'honest-10 variants done'
} else {
  HB "honest-10 regions missing ($HonestReg) - mock-only"
  "skipped no regions at $HonestReg" | Set-Content -Encoding ascii "$JobDir\out_honest10_skipped.txt"
}

# --- compliance on submission OUTPUT (tta_ms_prio) ---
$OnGeo = Join-Path $JobDir 'out_tta_ms_prio\result.geojson'
if (Test-Path $OnGeo) {
  $CName = 'koz3-bot8-ttabot3-compliance-' + (Get-Date -Format 'yyyyMMddHHmmss')
  $WtM3 = ($Wt -replace '\\','/')
  $OutM3 = ($JobDir -replace '\\','/')
  $ccArgs = @(
    'run','--rm','--name',$CName,'--network','none',
    '-e','PYTHONDONTWRITEBYTECODE=1','-e','PYTHONUNBUFFERED=1','-e','PYTHONIOENCODING=utf-8',
    '-v',"${WtM3}:/data/zip:ro",
    '-v',"${OutM3}:/data/output:ro",
    '-w','/data/zip', $Image,
    'python','scripts/check_compliance.py','/data/output/out_tta_ms_prio/result.geojson'
  )
  HB "docker compliance $CName"
  $p = Start-Process -FilePath 'docker' -ArgumentList $ccArgs -PassThru -NoNewWindow `
    -RedirectStandardOutput "$JobDir\compliance_tta_ms_prio.log" `
    -RedirectStandardError  "$JobDir\compliance_tta_ms_prio.err"
  Wait-Process -InputObject $p
  HB "compliance exit=$($p.ExitCode)"
  Set-Content -Encoding ascii "$JobDir\compliance_exit.txt" "$($p.ExitCode)"
} else {
  HB 'compliance skipped - no out_tta_ms_prio/result.geojson'
}

# --- score ---
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
if ($ScorePy -and $MetricPy -and (Test-Path $KitGt) -and (Test-Path "$JobDir\out_base_ae\result.geojson")) {
  $scoreArgs = @(
    $ScorePy, '--metric', $MetricPy, '--gt', $KitGt,
    '--a', "base_ae=$JobDir\out_base_ae\result.geojson"
  )
  foreach ($v in @('tta_ae','tta_ms_prio')) {
    $g = "$JobDir\out_$v\result.geojson"
    if (Test-Path $g) { $scoreArgs += @('--b', "$v=$g") }
  }
  $scoreArgs += @('--out', "$JobDir\scores.json")
  & $PyHost @scoreArgs *> "$JobDir\scores.log"
  if (Test-Path "$JobDir\scores.json") { HB 'scores.json written' } else { HB 'WARN score failed' }
} else {
  HB 'score skipped'
}

$need = @('out_base_ae','out_tta_ae','out_tta_ms_prio')
$missing = @($need | Where-Object { -not (Test-Path "$JobDir\$_\result.geojson") })
if ($missing.Count -gt 0) {
  Set-Status ("FAILED missing " + ($missing -join ','))
  exit 5
}
Set-Status 'DONE'
HB 'all done'
exit 0
