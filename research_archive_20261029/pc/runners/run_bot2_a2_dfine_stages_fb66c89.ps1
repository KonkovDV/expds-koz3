# bot2 A2 D-FINE dump+stages (A only; B reused) @ fb66c89 — FIRST GPU lane. Resumable. Detached-safe.
# A2: sp_dfine ON drop_groups=[] + EXPDS_DFINE_DUMP stages.json. B reused from hotfix. R forbidden.
$ErrorActionPreference = 'Continue'
[Console]::OutputEncoding = [Text.Encoding]::UTF8
$JobDir = 'C:\Sloi\pc_jobs\bot2_a2_dfine_stages_fb66c89'
$Wt = 'C:\Sloi\wt_bot2_fb66c89'
$Sha = 'fb66c899a6b534f6bbec06c7496736c3eaa4c3ac'
$Py = 'C:\Sloi\expds-koz3_v0.2.0\expds-koz3\.venv\Scripts\python.exe'
$Kit = 'C:\Sloi\pc_jobs\v3e2e'
$Regions = 'F:\Sloi_pc_jobs_archive\v3e2e\regions'   # junctions to mock sites (same as prior e2e)
$ZipV3 = 'F:\Sloi_pc_jobs_archive\v3e2e\kit\submission_bot7cand_v3on_0876090.zip'
$ZipV2 = 'F:\Sloi_pc_jobs_archive\v3e2e\kit\submission_bot7cand_v2_eb268d6.zip'
$ToolsTar = 'F:\Sloi_pc_jobs_archive\v3e2e\kit\tools_v3e2e.tar'
New-Item -ItemType Directory -Force -Path $JobDir, "$JobDir\out_A", "$JobDir\out_B", "$JobDir\notes" | Out-Null

function Set-Status([string]$s) {
  $line = "$s $(Get-Date -Format o)"
  Set-Content -Encoding UTF8 "$JobDir\STATUS.txt" -Value $line
  Add-Content -Encoding UTF8 "$JobDir\heartbeat.log" -Value $line
  Write-Host $line
}
function Heartbeat([string]$msg) { Add-Content -Encoding UTF8 "$JobDir\heartbeat.log" -Value "$(Get-Date -Format o) $msg" }

# Idempotent: skip stages already DONE
$st = if (Test-Path "$JobDir\STATUS.txt") { Get-Content "$JobDir\STATUS.txt" -Raw } else { '' }
if ($st -match '^DONE') { Write-Host 'already DONE'; exit 0 }

Set-Status 'RUNNING'
Heartbeat 'start'

# --- worktree ---
if (-not (Test-Path "$Wt\.git")) {
  Heartbeat 'worktree add'
  & cmd.exe /c "git -C C:\Sloi\wt_bots fetch origin bots/bot2-e2e-v3-prep" | Out-File -Encoding utf8 "$JobDir\git_fetch.log"
  & cmd.exe /c "git -C C:\Sloi\wt_bots worktree add --detach `"$Wt`" $Sha" 2>&1 | Out-File -Encoding utf8 "$JobDir\git_wt.log"
}
$head = (& cmd.exe /c "git -C `"$Wt`" rev-parse HEAD").Trim()
Heartbeat "worktree HEAD=$head"
if ($head -ne $Sha) { Set-Status "FAILED bad_sha $head"; exit 2 }

# --- tools / GT scorer (reuse kit) ---
$Tools = "$JobDir\tools"
if (-not (Test-Path "$Tools\e2e_score.py")) {
  New-Item -ItemType Directory -Force -Path $Tools | Out-Null
  tar -xf $ToolsTar -C $JobDir
  if (Test-Path "$JobDir\tools") { } else { tar -xf $ToolsTar -C $Tools }
}
# tools_v3e2e.tar usually extracts to tools/
$ScorePy = Get-ChildItem -Path $JobDir -Recurse -Filter e2e_score.py | Select-Object -First 1 -ExpandProperty FullName
$MetricPy = Get-ChildItem -Path $JobDir -Recurse -Filter compute_metrics.py | Select-Object -First 1 -ExpandProperty FullName
Heartbeat "scorer=$ScorePy metric=$MetricPy"

function Patch-Toml([string]$tomlPath, [ValidateSet('A','B')]$variant) {
  $t = Get-Content -Raw -Encoding UTF8 $tomlPath
  if ($variant -eq 'A') {
    # Ensure enabled=true and drop_groups=[] (never R). Strip R-style drop lists.
    $t = [regex]::Replace($t, '(?m)^\[sp_dfine\]\r?\nenabled = false\r?\n', "[sp_dfine]`r`nenabled = true`r`n")
    $t = [regex]::Replace($t, '(?m)^drop_groups = \[.*?\]\r?\n', "drop_groups = []`r`n")
    if ($t -notmatch '(?m)^drop_groups =') {
      $t = [regex]::Replace($t, '(?m)^(\[sp_dfine\]\r?\nenabled = true\r?\n)', "`$1drop_groups = []`r`n")
    }
  } else {
    # B: force enabled=false
    $t = [regex]::Replace($t, '(?m)^(\[sp_dfine\]\r?\n)enabled = true\r?\n', "`$1enabled = false`r`n")
    if ($t -notmatch '(?m)^\[sp_dfine\]\r?\nenabled = false') {
      $t = [regex]::Replace($t, '(?m)^\[sp_dfine\]\r?\n', "[sp_dfine]`r`nenabled = false`r`n")
    }
  }
  # Safety: never leave R drop_groups
  if ($t -match 'drop_groups = \["sp_or"') { throw "R drop_groups still present after patch $variant" }
  Set-Content -Encoding UTF8 -Path $tomlPath -Value $t -NoNewline
}

function Ensure-Unpacked([string]$tag, [string]$zip, [ValidateSet('A','B')]$variant) {
  $Z = "$JobDir\zip_$tag"
  if (-not (Test-Path "$Z\inference\main.py")) {
    New-Item -ItemType Directory -Force -Path $Z | Out-Null
    tar -xf $zip -C $Z 2>$null
    if (-not (Test-Path "$Z\inference\main.py")) {
      # zip may be Expand-Archive style
      Expand-Archive -Path $zip -DestinationPath $Z -Force
    }
  }
  $toml = "$Z\configs\default.toml"
  if (-not (Test-Path $toml)) { throw "no default.toml in $Z" }
  # Always re-patch (idempotent intent)
  Copy-Item -Force $toml "$toml.bak_orig" -ErrorAction SilentlyContinue
  # Prefer worktree pipeline/sp_dfine (safety valve @ fb66c89)
  foreach ($rel in @('expds\pipeline.py','expds\sp_dfine.py')) {
    $src = Join-Path $Wt $rel
    if (Test-Path $src) { Copy-Item -Force $src (Join-Path $Z $rel) }
  }
  Patch-Toml $toml $variant
  # Record which patch applied
  Select-String -Path $toml -Pattern '\[sp_dfine\]|enabled =|drop_groups' | Select-Object -First 12 | ForEach-Object { $_.Line } |
    Set-Content -Encoding UTF8 "$JobDir\toml_${tag}_snip.txt"
  return $Z
}

function Run-Main([string]$tag, [string]$Z, [string]$outGeo) {
  if ((Test-Path $outGeo) -and ((Get-Item $outGeo).Length -gt 1000)) {
    Heartbeat "skip main $tag (exists)"
    return 0
  }
  New-Item -ItemType Directory -Force -Path (Split-Path $outGeo) | Out-Null
  Heartbeat "main.py $tag start"
  $env:PYTHONDONTWRITEBYTECODE = '1'
  if ($tag -eq 'A') { $env:EXPDS_DFINE_DUMP = "$JobDir\dump_A" ; New-Item -ItemType Directory -Force -Path "$JobDir\dump_A" | Out-Null }
  else { Remove-Item Env:EXPDS_DFINE_DUMP -ErrorAction SilentlyContinue }
  $t1 = Get-Date
  $p = Start-Process -FilePath $Py -ArgumentList @("$Z\inference\main.py", "`"$Regions`"", "`"$outGeo`"") `
    -WorkingDirectory $Z -PassThru -NoNewWindow `
    -RedirectStandardOutput "$JobDir\main_$tag.log" -RedirectStandardError "$JobDir\main_$tag.err"
  while (-not $p.HasExited) {
    Heartbeat "main $tag running pid=$($p.Id) $([int]((Get-Date)-$t1).TotalSeconds)s"
    Start-Sleep -Seconds 20
  }
  $dt = ((Get-Date) - $t1).TotalSeconds
  Heartbeat "main $tag done exit=$($p.ExitCode) ${dt}s"
  return $p.ExitCode
}

# Prefer v3on zip as base for both (has D-FINE weights); B disables via toml.
$ZA = Ensure-Unpacked 'A' $ZipV3 'A'
$ZB = Ensure-Unpacked 'B' $ZipV3 'B'

$codeA = Run-Main 'A' $ZA "$JobDir\out_A\result.geojson"
# B: reuse prior hotfix B geojson if present (A≡B already known); else run B once
$PriorB = 'F:\Sloi_pc_jobs_archive\bot2_v3_hotfix_1ec2d7c\out_B\result.geojson'
if (Test-Path $PriorB) {
  New-Item -ItemType Directory -Force -Path "$JobDir\out_B" | Out-Null
  Copy-Item -Force $PriorB "$JobDir\out_B\result.geojson"
  Heartbeat 'reuse prior hotfix B geojson'
  $codeB = 0
} else {
  $codeB = Run-Main 'B' $ZB "$JobDir\out_B\result.geojson"
}
if ($codeA -ne 0 -or $codeB -ne 0) { Set-Status "FAILED main A=$codeA B=$codeB"; exit 3 }
# Copy stages dump note
if (Test-Path "$JobDir\dump_A\stages.json") { Copy-Item "$JobDir\dump_A\stages.json" "$JobDir\notes\stages.json" -Force }
Heartbeat "stages: $(Get-Content '$JobDir\dump_A\stages.json' -ErrorAction SilentlyContinue | Select-Object -First 5)"

# GT + e2e score (per-layer) — same as kit
$Gt = "$JobDir\gt.geojson"
if (-not (Test-Path $Gt)) {
  Push-Location (Split-Path $ScorePy | Join-Path -ChildPath 'repo' -ErrorAction SilentlyContinue)
  # tools layout: tools/repo or tools/
  $repoTools = Join-Path (Split-Path $ScorePy) 'repo'
  if (Test-Path "$repoTools\expds") {
    Push-Location $repoTools
    & $Py -m expds.gt --root $Regions --out $Gt *> "$JobDir\gt.log"
    Pop-Location
  } else {
    $env:PYTHONPATH = $Wt
    & $Py -c "from expds.gt import main" 2>$null
    & $Py -m expds.gt --root $Regions --out $Gt *> "$JobDir\gt.log"
  }
}
& $Py $ScorePy --metric $MetricPy --gt $Gt `
  --a "A=$JobDir\out_A\result.geojson" --b "B=$JobDir\out_B\result.geojson" `
  --out "$JobDir\e2e_AB.json" *> "$JobDir\e2e_AB.log"

# Also score vs v2 baseline if res_v2 exists from prior kit run
$V2 = 'F:\Sloi_pc_jobs_archive\v3e2e\out_v3fix2\res_v2\result.geojson'
if (-not (Test-Path $V2)) { $V2 = 'F:\Sloi_pc_jobs_archive\v3e2e\out\res_v2\result.geojson' }
if (Test-Path $V2) {
  & $Py $ScorePy --metric $MetricPy --gt $Gt `
    --a "v2=$V2" --b "A=$JobDir\out_A\result.geojson" --out "$JobDir\e2e_v2_A.json" *> "$JobDir\e2e_v2_A.log"
  & $Py $ScorePy --metric $MetricPy --gt $Gt `
    --a "v2=$V2" --b "B=$JobDir\out_B\result.geojson" --out "$JobDir\e2e_v2_B.json" *> "$JobDir\e2e_v2_B.log"
}

# honest-10 on work stack (match-scope if present on this branch)
$env:PYTHONPATH = $Wt
$hasLayer = Select-String -Path "$Wt\scripts\scoreboard.py" -Pattern 'match-scope' -Quiet
$scopeNote = if ($hasLayer) { 'layer' } else { 'global (match-scope not on branch)' }
$scopeNote | Set-Content "$JobDir\honest_scope.txt"
$Train = 'F:\2\train-expds-3'; $Work = 'F:\expds3\work'
$metricRepo = Join-Path $Wt 'materials\compute_metrics.py'
$honestArgs = @("$Wt\scripts\scoreboard.py", '--work', $Work, '--train', $Train, '--metric', $metricRepo, '--honest', '10', '--out', "$JobDir\notes\honest10.json")
if ($hasLayer) { $honestArgs += @('--match-scope', 'layer') }
Heartbeat "honest-10 scope=$scopeNote"
& $Py @honestArgs *> "$JobDir\honest10.log"

# LESSONS: do not mark DONE on path alone; require exit==0 from Run-Main stages
if ($null -ne (Get-Variable codeA -EA SilentlyContinue) -and $codeA -ne 0) { Set-Status "FAILED main_A_exit=$codeA"; exit 4 }
if ($null -ne (Get-Variable codeB -EA SilentlyContinue) -and $codeB -ne 0 -and -not (Test-Path "$JobDir\out_B\result.geojson")) { Set-Status "FAILED main_B_exit=$codeB"; exit 4 }
if (-not (Test-Path "$JobDir\out_A\result.geojson")) { Set-Status 'FAILED no_geojson_A'; exit 5 }
Set-Status 'DONE'
Heartbeat 'all done'
Get-Content "$JobDir\e2e_AB.json" -ErrorAction SilentlyContinue | Select-Object -First 80
