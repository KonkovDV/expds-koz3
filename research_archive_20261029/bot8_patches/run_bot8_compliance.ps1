# bot8 compliance + TTA unit tests (CPU). Paths guarded with Test-Path.
# NEVER pass centerhead/tests (absent -> pytest exit 4).
# COMPLIANCE TARGET = submission OUTPUT (result.geojson / LB submission zip),
# NEVER a code/weights kit zip (LESSONS section 5). Kit -> FAIL loud exit 2.
# JobDir/Wt on C: (F: full). ASCII-only (PS5 ANSI-safe). No python -c.
$ErrorActionPreference = 'Continue'
[Console]::OutputEncoding = [Text.Encoding]::UTF8

$Sha     = 'f6fe20a41b73e406c3a5ebd62134966e8c0bda6f'
if ($env:KOZ3_SHA) { $Sha = $env:KOZ3_SHA }
$Short   = $Sha.Substring(0,7)
$JobDir  = "C:\Sloi\pc_jobs\bot8_compliance_$Short"
$Wt      = 'C:\Sloi\pc_jobs\wt_bot8_tta_lb'
$Repo    = 'C:\Sloi\wt_bots'
$Py      = 'C:\Sloi\expds-koz3_v0.2.0\expds-koz3\.venv\Scripts\python.exe'
$Branch  = 'bots/bot8-tta-lb'

# Explicit submission output path (LB scores this, not the kit).
# Override: $env:KOZ3_SUBMISSION = full path to result.geojson or submission.zip
$Submission = $env:KOZ3_SUBMISSION
if (-not $Submission) {
  $cand = @(
    'C:\Sloi\pc_jobs\bot2_v2_base_docker_eb268d6\out_mock\result.geojson',
    'C:\Sloi\pc_jobs\bot2_v2_base_docker\out_mock\result.geojson',
    'C:\Sloi\pc_jobs\_lb_subs\submission_bot7cand_v2_eb268d6\result.geojson',
    'F:\Sloi_pc_jobs_archive\v3e2e\out\out_mock\result.geojson'
  )
  foreach ($c in $cand) {
    if (Test-Path $c) { $Submission = $c; break }
  }
}

New-Item -ItemType Directory -Force -Path $JobDir, "$JobDir\notes" | Out-Null
function HB($m){ Add-Content -Encoding utf8 "$JobDir\heartbeat.log" "$(Get-Date -Format o) $m"; Write-Host $m }
function Set-Status($s){
  $l = "$s $(Get-Date -Format o)"
  Set-Content -Encoding utf8 "$JobDir\STATUS.txt" $l
  HB $l
}

$st = if (Test-Path "$JobDir\STATUS.txt") { Get-Content "$JobDir\STATUS.txt" -Raw } else { '' }
if ($st -match '^DONE') { Write-Host 'already DONE'; exit 0 }

Set-Status 'RUNNING'
HB "start sha=$Sha wt=$Wt job=$JobDir submission=$Submission"

if (-not $Submission) {
  Set-Status 'FAILED no_KOZ3_SUBMISSION set env KOZ3_SUBMISSION to result.geojson or LB submission zip'
  exit 2
}
if (-not (Test-Path $Submission)) {
  Set-Status "FAILED submission_missing $Submission"
  exit 2
}

# Loud preflight: refuse kit paths by name/size before calling checker
$subName = [IO.Path]::GetFileName($Submission).ToLower()
$subLen  = (Get-Item $Submission).Length
if ($subName -match 'v3on|kit' -and $subName -match '\.zip$') {
  Set-Status "FAILED kit_path_refused name=$subName (LESSONS section 5: validate submission output not kit)"
  exit 2
}
if ($subLen -gt 100000000 -and $subName -match '\.zip$') {
  # >100 MB zip is almost certainly a kit; check_compliance will also reject
  HB "WARN submission zip size_mb=$([math]::Round($subLen/1e6,1)) (>100) - kit guard will decide"
}

if (-not (Test-Path (Join-Path $Wt '.git'))) {
  New-Item -ItemType Directory -Force -Path (Split-Path $Wt) | Out-Null
  & cmd.exe /c "git -C `"$Repo`" fetch origin $Branch" *> "$JobDir\notes\git_fetch.log"
  & cmd.exe /c "git -C `"$Repo`" worktree add --detach `"$Wt`" $Sha" *> "$JobDir\notes\git_wt.log"
} else {
  & cmd.exe /c "git -C `"$Repo`" fetch origin $Branch" *> "$JobDir\notes\git_fetch.log"
  & cmd.exe /c "git -C `"$Wt`" checkout --force --detach $Sha" *> "$JobDir\notes\git_wt.log"
}
$head = (& cmd.exe /c "git -C `"$Wt`" rev-parse HEAD").Trim()
HB "worktree HEAD=$head"
if ($head -ne $Sha) {
  Set-Status "FAILED bad_sha want=$Sha got=$head"
  exit 2
}

$pytestArgs = @('-q','--tb=line')
$candidates = @(
  'tests\test_check_compliance.py',
  'tests\test_tta.py'
)
$found = @()
foreach ($rel in $candidates) {
  $p = Join-Path $Wt $rel
  if (Test-Path $p) {
    $found += $rel
  } else {
    HB "MISSING $rel - skip (would be pytest exit 4)"
  }
}
if (Test-Path (Join-Path $Wt 'centerhead\tests')) {
  $found += 'centerhead\tests'
  HB 'centerhead/tests present - included'
} else {
  HB 'centerhead/tests absent - NOT passed to pytest (avoids exit 4)'
}

if ($found.Count -eq 0) {
  Set-Status 'FAILED no_test_files'
  exit 4
}

$env:PYTHONPATH = $Wt
$env:PYTHONIOENCODING = 'utf-8'
Push-Location $Wt
$pytestRc = 0
$ccRc = -1
try {
  HB ("pytest " + ($found -join ' '))
  & $Py -m pytest @pytestArgs @found *> "$JobDir\pytest.log"
  $pytestRc = $LASTEXITCODE
  HB "pytest exit=$pytestRc"
  Set-Content -Encoding utf8 "$JobDir\pytest_exit.txt" "$pytestRc"

  $cc = Join-Path $Wt 'scripts\check_compliance.py'
  if (-not (Test-Path $cc)) {
    Set-Status 'FAILED no_check_compliance_py'
    Pop-Location
    exit 2
  }
  HB "check_compliance TARGET=$Submission"
  Set-Content -Encoding utf8 "$JobDir\compliance_target.txt" $Submission
  & $Py $cc $Submission *> "$JobDir\compliance_submission.log"
  $ccRc = $LASTEXITCODE
  HB "check_compliance exit=$ccRc"
  Set-Content -Encoding utf8 "$JobDir\compliance_exit.txt" "$ccRc"
  if ($ccRc -eq 2) {
    # kit reject or read error - surface first lines
    Get-Content "$JobDir\compliance_submission.log" -EA SilentlyContinue |
      Select-Object -First 20 | ForEach-Object { HB "cc $_" }
  }
} catch {
  Set-Status "FAILED exception $_"
  Pop-Location
  exit 9
} finally {
  Pop-Location
}

if ($pytestRc -ne 0) {
  Set-Status "FAILED pytest_exit=$pytestRc"
  exit $pytestRc
}
if ($ccRc -ne 0) {
  Set-Status "FAILED compliance_exit=$ccRc target=$Submission"
  exit $ccRc
}
Set-Status 'DONE'
HB 'all done'
exit 0
