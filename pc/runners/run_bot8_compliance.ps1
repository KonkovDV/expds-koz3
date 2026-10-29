# bot8 compliance + TTA unit tests (CPU). Paths guarded with Test-Path.
# Never pass centerhead/tests (absent on bots/bot8-orsp-tta -> pytest exit 4).
# JobDir/Wt on C: (F: full). Unit pytest may use host venv.
# Gate e2e/honest-10 numbers must use Docker (PROTOCOL section 10).
# ASCII-only (PS5 ANSI-safe).
$ErrorActionPreference = 'Continue'
[Console]::OutputEncoding = [Text.Encoding]::UTF8

$Sha     = '82a167c7939d926b6da87228678a5adc51cdbdf1'
if ($env:KOZ3_SHA) { $Sha = $env:KOZ3_SHA }
$Short   = $Sha.Substring(0,7)
$JobDir  = "C:\Sloi\pc_jobs\bot8_compliance_$Short"
$Wt      = 'C:\Sloi\pc_jobs\wt_bot8_orsp_tta'
$Repo    = 'C:\Sloi\wt_bots'
$Py      = 'C:\Sloi\expds-koz3_v0.2.0\expds-koz3\.venv\Scripts\python.exe'
$Branch  = 'bots/bot8-orsp-tta'
# Inputs may still be READ from F:
$ZipV3   = 'F:\Sloi_pc_jobs_archive\v3e2e\kit\submission_bot7cand_v3on_0876090.zip'
if (-not (Test-Path $ZipV3)) {
  $ZipV3 = 'C:\Sloi\pc_jobs\_vendor_zips\submission_bot7cand_v3on_0876090.zip'
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
HB "start sha=$Sha wt=$Wt job=$JobDir"

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
if ($head -ne $Sha) {
  Set-Status "FAILED bad_sha want=$Sha got=$head"
  exit 2
}

# Build pytest argv: ONLY existing files. Never hard-code centerhead/tests.
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
try {
  HB ("pytest " + ($found -join ' '))
  & $Py -m pytest @pytestArgs @found *> "$JobDir\pytest.log"
  $pytestRc = $LASTEXITCODE
  HB "pytest exit=$pytestRc"
  Set-Content -Encoding utf8 "$JobDir\pytest_exit.txt" "$pytestRc"

  $cc = Join-Path $Wt 'scripts\check_compliance.py'
  if ((Test-Path $ZipV3) -and (Test-Path $cc)) {
    HB "check_compliance $ZipV3"
    & $Py $cc $ZipV3 *> "$JobDir\compliance_v3on.log"
    $ccRc = $LASTEXITCODE
    HB "check_compliance exit=$ccRc"
    Set-Content -Encoding utf8 "$JobDir\compliance_exit.txt" "$ccRc"
  } else {
    HB 'check_compliance skipped (no zip or script)'
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
Set-Status 'DONE'
HB 'all done'
exit 0
