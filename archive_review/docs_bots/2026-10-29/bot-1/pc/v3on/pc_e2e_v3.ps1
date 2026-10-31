# bot-7, 29.10.2026: СКВОЗНОЙ прогон v3 (флаги ВКЛ.: D-FINE R) против v2 eb268d6 на ПК (4090) — условие (2) бота 10 (REPORT10_accept §1).
# Читает и пишет ТОЛЬКО C:. F: не трогает. Ничего не посылает (ни LB, ни Kaggle, ни push, ни S3).
#   powershell -ExecutionPolicy Bypass -File C:\expds3c\bot7v3\pc_e2e_v3.ps1
# Что меряет: для каждого zip — main.py на ОДНОЙ папке регионов (как на платформе): время, пик рабочего набора дерева процессов,
# пик VRAM (nvidia-smi), затем официальная метрика v2 против v3 (всего / по регионам / по markup) по эталону из тех же регионов.
# ВАЖНО о честности: с весами f0 честны только регионы, которых f0 не видел (Гнёздово 101_Днепровская, 103_Ольшанская);
# с финальной моделью (всё SpOr) все регионы трейна — в выборке, такой прогон меряет только время/VRAM/работоспособность.
param(
  [string]$Py      = 'C:\Sloi\expds-koz3_v0.2.0\expds-koz3\.venv\Scripts\python.exe',  # ДОПУЩЕНИЕ: venv с torch+CUDA и пакетами образа
  [string]$Regions = 'C:\expds3c\regions_e2e',   # папка с регионами (подпапки NNN_Имя), напр. 101_Гнёздово_Днепровская, 103_Гнёздово_Ольшанская
  [string]$Bot7    = 'C:\expds3c\bot7v3',        # zip v2, zip v3on, tools_v3e2e.tar, SHA256SUMS
  [string]$ZipV2   = 'C:\expds3c\bot7v3\submission_bot7cand_v2_eb268d6.zip',
  [string]$ZipV3   = 'C:\expds3c\bot7v3\submission_bot7cand_v3on_TEST.zip',
  [string]$Out     = 'C:\expds3c\v3e2e'
)
$ErrorActionPreference = 'Stop'
# --- 0. только C:, входы, SHA256 ---
foreach ($p in @($Py,$Regions,$Bot7,$ZipV2,$ZipV3,$Out)) { if (-not $p.ToUpper().StartsWith('C:\')) { throw "не C:: $p" } }
foreach ($f in @($ZipV2,$ZipV3,"$Bot7\tools_v3e2e.tar","$Bot7\SHA256SUMS",$Regions)) { if (-not (Test-Path $f)) { throw "нет: $f" } }
$sums = @{}; Get-Content "$Bot7\SHA256SUMS" | ForEach-Object { $h,$n = $_ -split '\s+',2; $sums[$n.Trim()] = $h }
foreach ($f in @($ZipV2,$ZipV3,"$Bot7\tools_v3e2e.tar")) { $n = Split-Path $f -Leaf
  if (-not $sums.ContainsKey($n)) { throw "нет в SHA256SUMS: $n" }
  if ((Get-FileHash -Algorithm SHA256 $f).Hash.ToLower() -ne $sums[$n]) { throw "SHA256 не совпал: $n" } ; "sha ok $n" }
$env:PYTHONDONTWRITEBYTECODE = '1'
New-Item -ItemType Directory -Force $Out | Out-Null
if (-not (Test-Path "$Out\tools\e2e_score.py")) { tar -xf "$Bot7\tools_v3e2e.tar" -C $Out }
"start $(Get-Date -Format o)" | Tee-Object "$Out\time.txt"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv | Tee-Object -Append "$Out\time.txt"

function TreeWS([int]$root) {   # сумма рабочих наборов процесса и всех потомков (воркеры multiprocessing)
  $all = Get-CimInstance Win32_Process | Select-Object ProcessId,ParentProcessId,WorkingSetSize
  $ids = @($root); $i = 0
  while ($i -lt $ids.Count) { $ids += @($all | Where-Object { $_.ParentProcessId -eq $ids[$i] } | ForEach-Object { [int]$_.ProcessId }); $i++ }
  ($all | Where-Object { $ids -contains [int]$_.ProcessId } | Measure-Object WorkingSetSize -Sum).Sum }

# --- 1. main.py обоих zip на одной папке регионов (последовательно, GPU свободен) ---
foreach ($pair in @(@('v2',$ZipV2), @('v3',$ZipV3))) {
  $tag,$zip = $pair; $Z = "$Out\zip_$tag"
  if (-not (Test-Path "$Z\inference\main.py")) { Expand-Archive $zip -DestinationPath $Z }
  $o = "$Out\res_$tag\result.geojson"; New-Item -ItemType Directory -Force (Split-Path $o) | Out-Null
  $smi = Start-Process -FilePath nvidia-smi -ArgumentList @('--query-gpu=memory.used','--format=csv,noheader,nounits','-lms','500') `
         -RedirectStandardOutput "$Out\vram_$tag.txt" -PassThru -NoNewWindow
  $t1 = Get-Date
  $p = Start-Process -FilePath $Py -ArgumentList @("$Z\inference\main.py", "`"$Regions`"", "`"$o`"") -WorkingDirectory $Z -PassThru -NoNewWindow `
       -RedirectStandardOutput "$Out\main_$tag.log" -RedirectStandardError "$Out\main_$tag.err"
  $peak = 0; while (-not $p.HasExited) { try { $peak = [Math]::Max($peak, (TreeWS $p.Id)) } catch {}; Start-Sleep -Milliseconds 1000 }
  $dt = ((Get-Date) - $t1).TotalSeconds
  Stop-Process -Id $smi.Id -Force -ErrorAction SilentlyContinue; Start-Sleep 1
  $vr = (Get-Content "$Out\vram_$tag.txt" | Where-Object { $_ -match '^\s*\d+' } | ForEach-Object { [int]$_ } | Measure-Object -Maximum).Maximum
  "{0}: main.py {1:N1} c, пик WS дерева {2:N0} МБ, пик VRAM (вся карта) {3} МиБ, код {4}" -f $tag, $dt, ($peak/1MB), $vr, $p.ExitCode | Tee-Object -Append "$Out\time.txt"
  if ($p.ExitCode -ne 0) { "ВНИМАНИЕ: $tag код $($p.ExitCode), см. main_$tag.err" | Tee-Object -Append "$Out\time.txt" }
  Select-String -Path "$Out\main_$tag.log","$Out\main_$tag.err" -Pattern 'sp_dfine|D-FINE|дедлайн|deadline|watchdog' -ErrorAction SilentlyContinue |
    ForEach-Object { $_.Line } | Out-File -Encoding utf8 "$Out\dfine_lines_$tag.txt"
}
# --- 2. эталон по тем же регионам (код expds из tools, без слияния-подбора) и официальная метрика v2 против v3 ---
Push-Location "$Out\tools\repo"
try { & $Py -m expds.gt --root $Regions --out "$Out\gt.geojson" 2>&1 | Tee-Object "$Out\gt.log" } finally { Pop-Location }
& $Py "$Out\tools\e2e_score.py" --metric "$Out\tools\compute_metrics.py" --gt "$Out\gt.geojson" `
   --a "v2=$Out\res_v2\result.geojson" --b "v3=$Out\res_v3\result.geojson" --out "$Out\e2e.json" 2>&1 | Tee-Object "$Out\e2e.log"
"end $(Get-Date -Format o)" | Tee-Object -Append "$Out\time.txt"
# Отдать боту 7/10: time.txt, e2e.log, e2e.json, dfine_lines_v3.txt, main_v3.err (хвост). Порог: v3 ≤ 1200 с на полном публичном наборе
# меряет kernel бота 9 koz3-dfine-spor-final; здесь — время на этой папке и разница v3−v2 (цена D-FINE).
