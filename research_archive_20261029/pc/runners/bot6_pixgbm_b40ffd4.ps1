$ErrorActionPreference='Continue'
[Console]::OutputEncoding=[Text.Encoding]::UTF8
$JobDir='C:\Sloi\pc_jobs\bot6_pixgbm_b40ffd4'
$Wt='C:\Sloi\wt_bot6_b40ffd4'
$Py='C:\Sloi\expds-koz3_v0.2.0\expds-koz3\.venv\Scripts\python.exe'
$Root='F:\2\train-expds-3'
$Sha='b40ffd4'
$Branch='bots/bot6-pixgbm-cleanroom'
New-Item -ItemType Directory -Force -Path $JobDir | Out-Null
function HB($m){ Add-Content -Encoding utf8 "$JobDir\heartbeat.log" "$(Get-Date -Format o) $m" }
function Set-Status($s){ $l="$s $(Get-Date -Format o)"; Set-Content -Encoding utf8 "$JobDir\STATUS.txt" $l; HB $l }

Set-Status 'RUNNING'
HB "start PIXGBM cleanroom retrain threads=12"

# Ensure worktree
if (-not (Test-Path "$Wt\scripts\pixgbm_retrain_cleanroom.py")) {
  HB 'worktree missing — try create from main repo'
  $Repos = @(
    'C:\Sloi\expds-koz3_v0.2.0\expds-koz3',
    'C:\Sloi\repo',
    'C:\Sloi\expds-koz3',
    'C:\Sloi\wt_bots'
  )
  $main=$null
  foreach($r in $Repos){
    if (Test-Path "$r\.git") { $main=$r; break }
    if (Test-Path "$r\HEAD") { $main=$r; break } # bare
  }
  if (-not $main) {
    # reuse any existing wt as git dir source
    $any = Get-ChildItem C:\Sloi -Directory -Filter 'wt_bot*' | Select-Object -First 1
    if ($any) {
      $gitDir = git -C $any.FullName rev-parse --git-common-dir 2>$null
      if ($gitDir) { $main = (Resolve-Path (Join-Path $any.FullName $gitDir)).Path }
    }
  }
  HB "git_main=$main"
  if (-not $main) { Set-Status 'FAILED_NO_REPO'; exit 2 }
  git -C $main fetch origin $Branch 2>&1 | Tee-Object -FilePath "$JobDir\git_fetch.log" | Out-Null
  if (Test-Path $Wt) { Remove-Item -Recurse -Force $Wt -EA SilentlyContinue }
  git -C $main worktree remove --force $Wt 2>$null
  git -C $main worktree add --detach $Wt $Sha 2>&1 | Tee-Object -FilePath "$JobDir\git_wt.log"
  if (-not (Test-Path "$Wt\scripts\pixgbm_retrain_cleanroom.py")) { Set-Status 'FAILED_WT'; exit 3 }
}
HB "HEAD=$(git -C $Wt rev-parse --short HEAD)"

# lightgbm check
& $Py -c "import lightgbm as l; print(l.__version__)" 2>&1 | Tee-Object "$JobDir\lgbm_ver.txt"
HB "lgbm=$(Get-Content $JobDir\lgbm_ver.txt -Raw)"

$env:PIXGBM_NUM_THREADS='12'
$env:OMP_NUM_THREADS='12'
$env:OPENBLAS_NUM_THREADS='12'
$env:MKL_NUM_THREADS='12'
$env:PYTHONDONTWRITEBYTECODE='1'

$argsList = @(
  "$Wt\scripts\pixgbm_retrain_cleanroom.py",
  '--enable-retrain',
  '--root', $Root,
  '--out', $JobDir,
  '--num-threads', '12',
  '--expected-hashes', "$Wt\models\pixgbm\all\MODELS_NOT_IN_GIT.md"
)
$t1=Get-Date
$p=Start-Process -FilePath $Py -ArgumentList $argsList -WorkingDirectory $Wt -PassThru -NoNewWindow `
  -RedirectStandardOutput "$JobDir\retrain.out" -RedirectStandardError "$JobDir\retrain.err"
while(-not $p.HasExited){
  $sec=[int]((Get-Date)-$t1).TotalSeconds
  HB "retrain pid=$($p.Id) ${sec}s"
  Start-Sleep 30
}
$sec=[math]::Round(((Get-Date)-$t1).TotalSeconds,1)
HB "retrain done exit=$($p.ExitCode) sec=$sec"
Set-Content "$JobDir\wall_sec.txt" $sec

# Hashes of outputs
$hashPy = @'
import hashlib, json, pathlib
out = pathlib.Path(r"C:\Sloi\pc_jobs\bot6_pixgbm_b40ffd4")
exp = {
  "poly.txt": "d1fe3efafcd4377d41430d8828e881576c9024f9d7a5c89f55e23554e02ec800",
  "finds.txt": "5fa8c09d77d0914cc925fb6bb925a0ab1b64ccecf3c823d7f3546e71c4ceb04a",
}
# search common locations
cands = []
for pat in ["poly.txt","**/poly.txt","models/**/poly.txt"]:
  cands += list(out.glob(pat))
# also models subdirs
for sub in [out/"models", out/"models"/"all", out/"all"]:
  if (sub/"poly.txt").is_file(): cands.append(sub/"poly.txt")
seen=set(); files={}
for p in cands:
  p=p.resolve()
  if p in seen: continue
  seen.add(p)
  files[p.name]=p
# prefer models/all
poly = out/"models"/"all"/"poly.txt"
finds = out/"models"/"all"/"finds.txt"
if not poly.is_file():
  poly = next((p for p in out.rglob("poly.txt")), None)
if not finds.is_file():
  finds = next((p for p in out.rglob("finds.txt")), None)
res={"poly":None,"finds":None,"match":{}}
for name, path, key in [("poly.txt", poly, "poly"), ("finds.txt", finds, "finds")]:
  if path and path.is_file():
    h=hashlib.sha256(path.read_bytes()).hexdigest()
    res[key]={"path":str(path),"sha256":h,"size":path.stat().st_size,"match_expected": h==exp[name]}
    res["match"][name]=h==exp[name]
  else:
    res[key]={"missing":True}
    res["match"][name]=False
logp=out/"retrain_log.json"
res["retrain_log_exists"]=logp.is_file()
if logp.is_file():
  try: res["retrain_log"]=json.loads(logp.read_text(encoding="utf-8"))
  except Exception as e: res["retrain_log_err"]=str(e)
(out/"hash_compare.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
print(json.dumps(res, indent=2))
'@
Set-Content -Encoding utf8 "$JobDir\_hash_compare.py" $hashPy
& $Py "$JobDir\_hash_compare.py" *> "$JobDir\hash_compare.log"
HB 'hashes done'

if (Test-Path "$JobDir\retrain_log.json") {
  Set-Status 'DONE'
} else {
  # still DONE if models written
  $any = @(Get-ChildItem -Recurse $JobDir -Filter poly.txt -EA SilentlyContinue)
  if ($any) { Set-Status 'DONE_NO_LOG' } else { Set-Status 'FAILED' }
}
