# bot-7, 29.10.2026, ред. 2 (после НЕДЕЙСТВИТЕЛЬНОГО прогона бота 1: D-FINE не загрузился, v3 == v2 побайтно).
# СКВОЗНОЙ прогон v3on (D-FINE R) против v2 eb268d6 на ПК (4090) — условие (2) бота 10 (REPORT10_accept §1).
# Пишет и читает ТОЛЬКО C:. F: не трогает. Ничего не посылает (ни LB, ни Kaggle, ни push, ни S3).
#   режим docker (ПРЕДПОЧТИТЕЛЬНО — как площадка: linux python 3.12, вендор колёс архива работает как есть):
#     powershell -ExecutionPolicy Bypass -File pc_e2e_v3.ps1 -Mode docker -Image pantum/n7q:20261024
#   режим venv (нативный Windows python 3.12): вендор архива (linux .so regex/tokenizers) на Windows НЕ грузится,
#     поэтому transformers 5.21.0 ставится ОТДЕЛЬНО в $TfDir (pip --target --no-deps, venv не меняется) и подаётся
#     через PYTHONPATH только процессу v3; sp_dfine примет его, только если версия ровно 5.21.0:
#     powershell -ExecutionPolicy Bypass -File pc_e2e_v3.ps1 -Mode venv -SetupTf
# Прогон признаётся ДЕЙСТВИТЕЛЬНЫМ, только если: проба импорта ok; в логе v3 есть «sp_dfine n_det=N total» с N>0;
# нет строк SP_DFINE_UNAVAILABLE / GOR_GATE_UNAVAILABLE; result v3 != result v2.
param(
  [ValidateSet('docker','venv')][string]$Mode = 'docker',
  [string]$Image   = 'pantum/n7q:20261024',
  [string]$Py      = 'C:\Sloi\expds-koz3_v0.2.0\expds-koz3\.venv\Scripts\python.exe',  # venv с torch+CUDA (режим venv и шаг 2)
  [string]$TfDir   = 'C:\Sloi\pc_jobs\v3e2e\tf521',     # только режим venv: отдельный каталог transformers 5.21.0
  [switch]$SetupTf,
  [string]$Regions = 'C:\Sloi\pc_jobs\v3e2e\regions',   # подпапки NNN_Имя: 101_…Днепровская, 103_…Ольшанская — КОПИИ на C:, не junction на F:
  [string]$Bot7    = 'C:\Sloi\pc_jobs\v3e2e\kit',
  [string]$ZipV2   = 'C:\Sloi\pc_jobs\v3e2e\kit\submission_bot7cand_v2_eb268d6.zip',
  [string]$ZipV3   = 'C:\Sloi\pc_jobs\v3e2e\kit\submission_bot7cand_v3on_36af6b7.zip',
  [string]$Out     = 'C:\Sloi\pc_jobs\v3e2e\out2'
)
$ErrorActionPreference = 'Stop'
function Native([scriptblock]$b) { $old = $ErrorActionPreference; $ErrorActionPreference = 'Continue'; try { & $b } finally { $ErrorActionPreference = $old } }
# --- 0. только C:, входы, SHA256 ---
foreach ($p in @($Py,$TfDir,$Regions,$Bot7,$ZipV2,$ZipV3,$Out)) { if (-not $p.ToUpper().StartsWith('C:\')) { throw "не C:: $p" } }
Get-ChildItem $Regions -Directory | ForEach-Object { if ($_.LinkType) { throw "регион $($_.Name) — $($_.LinkType) (ссылка, вероятно на F:): скопируйте на C:" } }
foreach ($f in @($ZipV2,$ZipV3,"$Bot7\tools_v3e2e.tar","$Bot7\SHA256SUMS",$Regions)) { if (-not (Test-Path $f)) { throw "нет: $f" } }
$sums = @{}; Get-Content "$Bot7\SHA256SUMS" | ForEach-Object { $h,$n = $_ -split '\s+',2; $sums[$n.Trim().TrimStart('*')] = $h }
foreach ($f in @($ZipV2,$ZipV3,"$Bot7\tools_v3e2e.tar")) { $n = Split-Path $f -Leaf
  if (-not $sums.ContainsKey($n)) { throw "нет в SHA256SUMS: $n" }
  if ((Get-FileHash -Algorithm SHA256 $f).Hash.ToLower() -ne $sums[$n]) { throw "SHA256 не совпал: $n" } ; "sha ok $n" }
$env:PYTHONDONTWRITEBYTECODE = '1'
New-Item -ItemType Directory -Force $Out | Out-Null
if (-not (Test-Path "$Out\tools\e2e_score.py")) { tar -xf "$Bot7\tools_v3e2e.tar" -C $Out }
"start $(Get-Date -Format o) mode=$Mode" | Tee-Object "$Out\time.txt"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv | Tee-Object -Append "$Out\time.txt"
foreach ($pair in @(@('v2',$ZipV2), @('v3',$ZipV3))) { $tag,$zip = $pair; $Z = "$Out\zip_$tag"
  if (-not (Test-Path "$Z\inference\main.py")) { if ((Get-PSDrive C).Free -lt 2.5GB) { throw "C: < 2.5 GB" }
    New-Item -ItemType Directory -Force $Z | Out-Null; tar -xf $zip -C $Z } }
if ((Select-String -Path "$Out\zip_v3\configs\default.toml" -Pattern '^\[sp_dfine\]' -Context 0,1).Context.PostContext -notmatch 'enabled = true') { throw "в zip v3 [sp_dfine] не enabled" }
if (-not (Test-Path "$Out\zip_v3\models\dfine_f0\model.safetensors")) { throw "в zip v3 нет весов models/dfine_f0" }

# --- 0b. (venv) отдельный transformers 5.21.0 под Windows: те же 10 пакетов, что в вендоре, без зависимостей ---
if ($Mode -eq 'venv' -and $SetupTf) {
  Native { & $Py -m pip install --no-deps --target $TfDir transformers==5.21.0 tokenizers==0.23.3 regex==2026.9.29 typer==0.27.3 `
      rich==15.0.0 markdown-it-py==4.2.0 mdurl==0.1.2 pygments==2.21.0 shellingham==1.5.4 annotated-doc==0.0.5 2>&1 | Tee-Object "$Out\setup_tf.log" }
  if ($LASTEXITCODE -ne 0) { throw "pip --target не удался, см. setup_tf.log" } }

# --- 0c. проба импорта D-FINE в ЧИСТОМ процессе (тот же код, что check_submission DFINE_PROBE) — до траты GPU ---
@'
import json, sys
root = sys.argv[1]; sys.path.insert(0, root)
try:
    from expds.sp_dfine import import_dfine
    C, M, info = import_dfine(root)
    out = {"ok": True, **{k: info.get(k) for k in ("source", "version", "file", "errors", "purged_preloaded")}}
except Exception as exc:
    out = {"ok": False, "err": f"{type(exc).__name__}: {str(exc)[:1500]}"}
print("DFINE_JSON=" + json.dumps(out, ensure_ascii=False))
'@ | Set-Content -Encoding utf8 "$Out\dfine_probe.py"
if ($Mode -eq 'docker') {
  $pr = Native { docker run --rm --network none -v "$Out\zip_v3:/app:ro" -v "${Out}:/probe:ro" -w /app $Image python -s /probe/dfine_probe.py /app 2>&1 }
} else {
  $env:PYTHONPATH = $TfDir; $pr = Native { & $Py -s "$Out\dfine_probe.py" "$Out\zip_v3" 2>&1 }; Remove-Item Env:PYTHONPATH
}
$pr | Out-File -Encoding utf8 "$Out\dfine_probe.txt"; $pj = ($pr | Where-Object { "$_" -like 'DFINE_JSON=*' } | Select-Object -Last 1)
"проба: $pj" | Tee-Object -Append "$Out\time.txt"
if (-not $pj -or ($pj -replace '^DFINE_JSON=','' | ConvertFrom-Json).ok -ne $true) { throw "D-FINE не импортируется в режиме $Mode — прогон бессмыслен, см. dfine_probe.txt" }

function TreeWS([int]$root) {
  $all = Get-CimInstance Win32_Process | Select-Object ProcessId,ParentProcessId,WorkingSetSize
  $ids = @($root); $i = 0
  while ($i -lt $ids.Count) { $ids += @($all | Where-Object { $_.ParentProcessId -eq $ids[$i] } | ForEach-Object { [int]$_.ProcessId }); $i++ }
  ($all | Where-Object { $ids -contains [int]$_.ProcessId } | Measure-Object WorkingSetSize -Sum).Sum }

# --- 1. main.py обоих zip на одной папке регионов (последовательно, GPU свободен), v2 и v3 в ОДНОМ режиме ---
foreach ($tag in @('v2','v3')) {
  $Z = "$Out\zip_$tag"; $od = "$Out\res_$tag"; New-Item -ItemType Directory -Force $od | Out-Null; Remove-Item -Force -ErrorAction SilentlyContinue "$od\result.geojson"
  $smi = Start-Process -FilePath nvidia-smi -ArgumentList @('--query-gpu=memory.used','--format=csv,noheader,nounits','-lms','500') `
         -RedirectStandardOutput "$Out\vram_$tag.txt" -PassThru -NoNewWindow
  $t1 = Get-Date
  if ($Mode -eq 'docker') {
    $args_ = @('run','--rm','--gpus','all','--network','none','--read-only','--user','12345:12345','--memory','96g','--pids-limit','8192','--shm-size','16g',
               '-v',"${Z}:/app:ro",'-v',"${Regions}:/data/input:ro",'-v',"${od}:/data/output",'-w','/app',$Image,'python','inference/main.py','/data/input','/data/output/result.geojson')
    $p = Start-Process -FilePath docker -ArgumentList $args_ -PassThru -NoNewWindow -RedirectStandardOutput "$Out\main_$tag.log" -RedirectStandardError "$Out\main_$tag.err"
  } else {
    if ($tag -eq 'v3') { $env:PYTHONPATH = $TfDir }
    $p = Start-Process -FilePath $Py -ArgumentList @("$Z\inference\main.py", "`"$Regions`"", "`"$od\result.geojson`"") -WorkingDirectory $Z -PassThru -NoNewWindow `
         -RedirectStandardOutput "$Out\main_$tag.log" -RedirectStandardError "$Out\main_$tag.err"
    Remove-Item -ErrorAction SilentlyContinue Env:PYTHONPATH
  }
  $peak = 0; while (-not $p.HasExited) { if ($Mode -eq 'venv') { try { $peak = [Math]::Max($peak, (TreeWS $p.Id)) } catch {} }; Start-Sleep -Milliseconds 1000 }
  $dt = ((Get-Date) - $t1).TotalSeconds
  Stop-Process -Id $smi.Id -Force -ErrorAction SilentlyContinue; Start-Sleep 1
  $vr = (Get-Content "$Out\vram_$tag.txt" | Where-Object { $_ -match '^\s*\d+' } | ForEach-Object { [int]$_ } | Measure-Object -Maximum).Maximum
  "{0}: main.py {1:N1} c, пик WS дерева {2:N0} МБ (только venv), пик VRAM (вся карта) {3} МиБ, код {4}" -f $tag, $dt, ($peak/1MB), $vr, $p.ExitCode | Tee-Object -Append "$Out\time.txt"
  Select-String -Path "$Out\main_$tag.log","$Out\main_$tag.err" -Pattern 'sp_dfine|D-FINE|UNAVAILABLE|transformers|дедлайн|deadline|watchdog' -ErrorAction SilentlyContinue |
    ForEach-Object { $_.Line } | Out-File -Encoding utf8 "$Out\dfine_lines_$tag.txt"
}
# --- 1b. действительность прогона v3 ---
$bad = @()
$tot = Select-String -Path "$Out\dfine_lines_v3.txt" -Pattern 'sp_dfine n_det=(\d+) total' | Select-Object -Last 1
if (-not $tot) { $bad += 'нет строки sp_dfine n_det=… total' } elseif ([int]$tot.Matches[0].Groups[1].Value -le 0) { $bad += 'sp_dfine n_det total = 0' }
if (Select-String -Path "$Out\dfine_lines_v3.txt" -Pattern 'SP_DFINE_UNAVAILABLE|GOR_GATE_UNAVAILABLE' -Quiet) { $bad += 'есть строка *_UNAVAILABLE' }
if ((Get-FileHash "$Out\res_v2\result.geojson").Hash -eq (Get-FileHash "$Out\res_v3\result.geojson").Hash) { $bad += 'result v3 == v2 побайтно' }
if ($bad) { "ПРОГОН НЕДЕЙСТВИТЕЛЕН: $($bad -join '; ')" | Tee-Object -Append "$Out\time.txt" } else { "прогон v3 действителен: $($tot.Line)" | Tee-Object -Append "$Out\time.txt" }
# --- 2. эталон по тем же регионам и официальная метрика v2 против v3 (stderr INFO не валит скрипт: Native) ---
Push-Location "$Out\tools\repo"
try { Native { & $Py -m expds.gt --root $Regions --out "$Out\gt.geojson" 2>&1 | ForEach-Object { "$_" } | Tee-Object "$Out\gt.log" } } finally { Pop-Location }
if (-not (Test-Path "$Out\gt.geojson")) { throw "эталон не построен, см. gt.log" }
Native { & $Py "$Out\tools\e2e_score.py" --metric "$Out\tools\compute_metrics.py" --gt "$Out\gt.geojson" `
   --a "v2=$Out\res_v2\result.geojson" --b "v3=$Out\res_v3\result.geojson" --out "$Out\e2e.json" 2>&1 | ForEach-Object { "$_" } | Tee-Object "$Out\e2e.log" }
"end $(Get-Date -Format o)" | Tee-Object -Append "$Out\time.txt"
"DONE $(Get-Date -Format o)" | Set-Content "$Out\..\STATUS.txt"
# Отдать боту 7/10: time.txt, dfine_probe.txt, dfine_lines_v3.txt, e2e.log, e2e.json, хвост main_v3.err.
