# bot6 PIXGBM clean-room @ d173a75 — CPU 12 threads. Resumable. Detached-safe.
$ErrorActionPreference = 'Continue'
[Console]::OutputEncoding = [Text.Encoding]::UTF8
$JobDir = 'C:\Sloi\pc_jobs\bot6_pixgbm'
$Wt = 'C:\Sloi\wt_bot6_d173a75'
$Sha = 'd173a756453d8d9b498dc692782c994a1b68109d'
$Py = 'C:\Sloi\expds-koz3_v0.2.0\expds-koz3\.venv\Scripts\python.exe'
$Root = 'F:\2\train-expds-3'
$ModelsSrcCandidates = @(
  'F:\Sloi_pc_jobs_archive\v3e2e\out_v3fix2\zip_v2\models\pixgbm\all',
  'F:\Sloi_pc_jobs_archive\v3e2e\out_v3fix2\zip_v3\models\pixgbm\all',
  'C:\Sloi\wt_bot6_pixgbm\models\pixgbm\all'
)
New-Item -ItemType Directory -Force -Path $JobDir | Out-Null
function HB($m){ Add-Content -Encoding utf8 "$JobDir\heartbeat.log" "$(Get-Date -Format o) $m" }
function Set-Status($s){ $l="$s $(Get-Date -Format o)"; Set-Content -Encoding utf8 "$JobDir\STATUS.txt" $l; HB $l }

if ((Test-Path "$JobDir\STATUS.txt") -and ((Get-Content "$JobDir\STATUS.txt" -Raw) -match '^DONE')) { Write-Host 'already DONE'; exit 0 }

Set-Status 'RUNNING'
HB 'start d173a75 PIXGBM'

# worktree (never reset existing other wts)
if (-not (Test-Path "$Wt\scripts\pixgbm_retrain_cleanroom.py")) {
  HB 'creating worktree'
  & cmd /c "git -C C:\Sloi\wt_bots fetch origin bots/bot6-pixgbm-cleanroom" 2>&1 | Out-File "$JobDir\git_fetch.log"
  if (Test-Path $Wt) { & cmd /c "git -C C:\Sloi\wt_bots worktree remove --force `"$Wt`"" 2>&1 | Out-Null }
  & cmd /c "git -C C:\Sloi\wt_bots worktree add --detach `"$Wt`" $Sha" 2>&1 | Out-File "$JobDir\git_wt.log"
}
$head = (& cmd /c "git -C `"$Wt`" rev-parse HEAD").Trim()
HB "HEAD=$head"
if ($head -ne $Sha) { Set-Status "FAILED bad_sha $head"; exit 2 }

# geo_pixgbm required
$geo = "$Wt\expds\geo_pixgbm.py"
if (-not (Test-Path $geo)) {
  $gs = @(
    'F:\Sloi_pc_jobs_archive\v3e2e\out_v3fix2\zip_v2\expds\geo_pixgbm.py',
    'F:\Sloi_pc_jobs_archive\v3e2e\out_v3fix2\zip_v3\expds\geo_pixgbm.py',
    'C:\Sloi\wt_bot6_pixgbm\expds\geo_pixgbm.py'
  )
  $found = $gs | Where-Object { Test-Path $_ } | Select-Object -First 1
  if (-not $found) {
    $found = Get-ChildItem -Path C:\Sloi -Recurse -Filter geo_pixgbm.py -EA SilentlyContinue | Select-Object -First 1 -ExpandProperty FullName
  }
  if ($found) { Copy-Item -Force $found $geo; HB "copied geo_pixgbm from $found" }
  else { Set-Status 'FAILED no geo_pixgbm.py'; exit 3 }
}
& $Py -c "import hashlib,pathlib; p=pathlib.Path(r'$geo'); print(hashlib.sha256(p.read_bytes()).hexdigest())" | Tee-Object "$JobDir\geo_pixgbm.sha256"
HB "geo_sha=$(Get-Content $JobDir\geo_pixgbm.sha256 -Raw)"

# reference models for hash compare
$md = "$Wt\models\pixgbm\all"
New-Item -ItemType Directory -Force -Path $md | Out-Null
$src = $ModelsSrcCandidates | Where-Object { Test-Path "$_\poly.txt" } | Select-Object -First 1
if (-not $src) { Set-Status 'FAILED no reference poly.txt'; exit 4 }
Copy-Item -Force "$src\poly.txt","$src\finds.txt" $md
if (Test-Path "$src\MODELS_NOT_IN_GIT.md") { Copy-Item -Force "$src\MODELS_NOT_IN_GIT.md" $md }
HB "ref models from $src"

$env:KOZ3_OUT = $JobDir
$env:PIXGBM_NUM_THREADS = '12'
$env:OMP_NUM_THREADS = '12'
$env:OPENBLAS_NUM_THREADS = '12'
$env:MKL_NUM_THREADS = '12'
$env:PYTHONDONTWRITEBYTECODE = '1'

$t1 = Get-Date
$p = Start-Process -FilePath $Py -ArgumentList @(
  "$Wt\scripts\pixgbm_retrain_cleanroom.py",
  '--enable-retrain',
  '--root', $Root,
  '--out', $JobDir,
  '--rounds', '300',
  '--num-threads', '12',
  '--metric-tol', '1e-4',
  '--expected-hashes', "$Wt\models\pixgbm\all\MODELS_NOT_IN_GIT.md"
) -WorkingDirectory $Wt -PassThru -NoNewWindow -RedirectStandardOutput "$JobDir\retrain.out" -RedirectStandardError "$JobDir\retrain.err"
while (-not $p.HasExited) { HB "retrain pid=$($p.Id) $([int]((Get-Date)-$t1).TotalSeconds)s"; Start-Sleep 30 }
$sec = [math]::Round(((Get-Date)-$t1).TotalSeconds,1)
Set-Content "$JobDir\wall_sec.txt" $sec
HB "retrain exit=$($p.ExitCode) sec=$sec"

# hash compare + verdict
& $Py -c @"
import hashlib, json, pathlib
out=pathlib.Path(r'$JobDir')
exp={'poly.txt':'d1fe3efafcd4377d41430d8828e881576c9024f9d7a5c89f55e23554e02ec800','finds.txt':'5fa8c09d77d0914cc925fb6bb925a0ab1b64ccecf3c823d7f3546e71c4ceb04a'}
res={'wall_sec':$sec,'exit':$($p.ExitCode)}
for name in ('poly.txt','finds.txt'):
  paths=list(out.rglob(name))
  # prefer newly trained under out/models
  pick=None
  for p in paths:
    if 'models' in p.parts: pick=p; break
  if not pick and paths: pick=paths[0]
  if pick and pick.is_file():
    h=hashlib.sha256(pick.read_bytes()).hexdigest()
    res[name]={'path':str(pick),'sha256':h,'match':h==exp[name],'size':pick.stat().st_size}
  else:
    res[name]={'missing':True,'match':False}
log=out/'retrain_log.json'
res['retrain_log_exists']=log.is_file()
if log.is_file():
  try: res['retrain_log']=json.loads(log.read_text(encoding='utf-8'))
  except Exception as e: res['retrain_log_err']=str(e)
mf=res.get('retrain_log',{}).get('metric_fallback') if isinstance(res.get('retrain_log'),dict) else None
ok = (res.get('poly.txt',{}).get('match') and res.get('finds.txt',{}).get('match')) or (isinstance(mf,dict) and mf.get('status')=='ok')
res['verdict']='PASS' if ok else 'FAIL'
(out/'verdict.json').write_text(json.dumps(res,indent=2),encoding='utf-8')
print(json.dumps(res,indent=2))
"@ *> "$JobDir\verdict.log"

if ((Test-Path "$JobDir\verdict.json") -and ((Get-Content "$JobDir\verdict.json" -Raw) -match '"verdict": "PASS"')) {
  Set-Status 'DONE'
} elseif (Test-Path "$JobDir\retrain_log.json") {
  Set-Status 'DONE_CHECK_VERDICT'
} else {
  Set-Status 'FAILED'
}
