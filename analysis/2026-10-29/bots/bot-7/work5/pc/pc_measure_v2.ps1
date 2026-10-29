# bot-7, 29.10.2026: трёхсторонний честный перемер кандидата v2 НА ПК: {S только, stack6 только, оба pix→stack6} против базы A.
# Читает и пишет ТОЛЬКО C:. Ничего не посылает (ни LB, ни Kaggle, ни push). Код замера — из tools_v2.tar (самодостаточный:
# bot-10 repo 68710a9 + work4/{stack6_fam,paired}.py, expds ветки bot7-candidate-v2 @b1a9d28 = af97af0 по expds/, скрипты bot-7).
#   powershell -ExecutionPolicy Bypass -File C:\expds3c\bot7v2\pc_measure_v2.ps1
#   + замер main.py кандидата на участке:  ... -MainRegion C:\expds3c\regions\<участок> -Zip C:\expds3c\bot7v2\submission_bot7cand_v2_af97af0.zip
param(
  [string]$Py   = 'C:\Sloi\expds-koz3_v0.2.0\expds-koz3\.venv\Scripts\python.exe',  # ДОПУЩЕНИЕ: venv с numpy/shapely/geopandas (как в pc_measure_pkg.ps1)
  [string]$In   = 'C:\expds3c\in',      # stack_base.geojson, gt_train_v2.geojson, pix_lofo.geojson
  [string]$Bot7 = 'C:\expds3c\bot7v2',  # tools_v2.tar, SHA256SUMS, family_clusters.json, [zip]
  [string]$Out  = 'C:\expds3c\v2meas',
  [int]$Boot    = 2000,
  [string]$MainRegion = '',
  [string]$Zip  = ''
)
$ErrorActionPreference = 'Stop'
# --- 0. только C:, входы, SHA256 ---
foreach ($p in @($Py,$In,$Bot7,$Out) + @($MainRegion,$Zip | Where-Object { $_ })) {
  if (-not $p.ToUpper().StartsWith('C:\')) { throw "не C:: $p" } }
$need = @("$In\stack_base.geojson", "$In\gt_train_v2.geojson", "$In\pix_lofo.geojson",
          "$Bot7\tools_v2.tar", "$Bot7\family_clusters.json", "$Bot7\SHA256SUMS")
foreach ($f in $need) { if (-not (Test-Path $f)) { throw "нет файла: $f (см. REPORT4 §7)" } }
$sums = @{}; Get-Content "$Bot7\SHA256SUMS" | ForEach-Object { $h,$n = $_ -split '\s+',2; $sums[$n.Trim()] = $h }
foreach ($f in @("$In\stack_base.geojson","$In\gt_train_v2.geojson","$In\pix_lofo.geojson","$Bot7\tools_v2.tar","$Bot7\family_clusters.json") + @($Zip | Where-Object { $_ })) {
  $n = Split-Path $f -Leaf
  if ($sums.ContainsKey($n)) { $h = (Get-FileHash -Algorithm SHA256 $f).Hash.ToLower()
    if ($h -ne $sums[$n]) { throw "SHA256 не совпал: $n" } else { "sha ok $n" } } else { "sha нет в списке: $n (пропуск)" } }
$env:OMP_NUM_THREADS = '2'; $env:MKL_NUM_THREADS = '2'; $env:PYTHONDONTWRITEBYTECODE = '1'; $env:EXPDS_MATCH_SCOPE = 'global'
New-Item -ItemType Directory -Force $Out | Out-Null
$T = "$Out\tools"
if (-not (Test-Path "$T\bot7\geo_exact.py")) { tar -xf "$Bot7\tools_v2.tar" -C $Out }   # tar.exe есть в Windows 10+
$env:EXPDS_METRIC = "$T\b10\repo\materials\compute_metrics.py"
$GT = "$In\gt_train_v2.geojson"; $CL = "$Bot7\family_clusters.json"
$t0 = Get-Date
# --- 1. посемейные стеки: A (база), S (pix_lofo вместо Geo-источника на 3 семьях), s6 и Ss6 (stack6 separate; Ss6 = pix→stack6) ---
if (-not (Test-Path "$Out\fam_A\_counts.json")) { & $Py "$T\b10\repo\scripts\split_stack_by_family.py" --stack "$In\stack_base.geojson" --out "$Out\fam_A" }
if (-not (Test-Path "$Out\fam_S")) { & $Py "$T\bot7\mkfam_pix_v2.py" --src "$Out\fam_A" --pix "$In\pix_lofo.geojson" --out "$Out\fam_S" }
if (-not (Test-Path "$Out\fam_s6"))  { & $Py "$T\b10\work4\stack6_fam.py" --src "$Out\fam_A" --dst "$Out\fam_s6"  --mode separate }
if (-not (Test-Path "$Out\fam_Ss6")) { & $Py "$T\b10\work4\stack6_fam.py" --src "$Out\fam_S" --dst "$Out\fam_Ss6" --mode separate }
# --- 2. кривые: A — все семьи; S/s6/Ss6 — копия кривых A + пересчёт 3 семей (Гнёздово, Семибратнее, Судьбищи) ---
if (-not (Test-Path "$Out\curves_A")) {
  & $Py "$T\b10\repo\scripts\family_curves.py" --fam "$Out\fam_A" --gt $GT --clusters $CL --out "$Out\curves_A" 2>&1 | Tee-Object "$Out\curves_A.log" }
# ИСПРАВЛЕНО (по замечанию бота 1): fam_S/s6/Ss6 содержат только 3 изменённые семьи, а family_curves.py пропускает существующие pkl.
# Теперь: (1) в fam_<v> докладываем недостающие jsonl из fam_A (чтобы кластеры-единицы «A+B» считались по полному составу);
# (2) pkl из curves_A копируем ТОЛЬКО для единиц, у которых все члены побайтно совпадают с fam_A; остальные пересчитываются.
# Старые curves_* без маркера _done_v2 (от прошлого ошибочного прогона) удаляются.
foreach ($v in @('S','s6','Ss6')) { if (-not (Test-Path "$Out\curves_$v\_done_v2")) {
  foreach ($j in Get-ChildItem "$Out\fam_A\*.jsonl") { $jv = Join-Path "$Out\fam_$v" $j.Name
    if (-not (Test-Path -LiteralPath $jv)) { Copy-Item -LiteralPath $j.FullName $jv } }
  if (Test-Path "$Out\curves_$v") { Remove-Item -Recurse -Force "$Out\curves_$v" }
  New-Item -ItemType Directory -Force "$Out\curves_$v" | Out-Null
  $changed = @()
  foreach ($pk in Get-ChildItem "$Out\curves_A\*.pkl") {
    $same = $true
    foreach ($m in ($pk.BaseName -split '\+')) { $ja = "$Out\fam_A\$m.jsonl"; $jv = "$Out\fam_$v\$m.jsonl"
      $ea = Test-Path -LiteralPath $ja; $ev = Test-Path -LiteralPath $jv
      if ($ea -ne $ev) { $same = $false }
      elseif ($ea -and ((Get-FileHash -Algorithm SHA256 -LiteralPath $ja).Hash -ne (Get-FileHash -Algorithm SHA256 -LiteralPath $jv).Hash)) { $same = $false } }
    if ($same) { Copy-Item -LiteralPath $pk.FullName "$Out\curves_$v\" -Force } else { $changed += $pk.BaseName } }
  "curves_${v}: пересчёт ($($changed.Count)): $($changed -join ', ')" | Tee-Object -Append "$Out\changed_families.txt"
  if ($changed.Count -eq 0) { throw "curves_${v}: ни одна единица не отличается от A — проверьте fam_$v" }
  & $Py "$T\b10\repo\scripts\family_curves.py" --fam "$Out\fam_$v" --gt $GT --clusters $CL --out "$Out\curves_$v" 2>&1 | Tee-Object "$Out\curves_$v.log"
  New-Item "$Out\curves_$v\_done_v2" -ItemType File | Out-Null } }
# --- 3. парный бутстрэп bot-10 (порядок пост-обработки bot-10: stack6 на сырых стеках до порогов) ---
Push-Location "$T\b10"   # paired.py берёт repo/ от CWD
try {
  & $Py work4\paired.py --boot $Boot --base "A=$Out\curves_A" --b "S=$Out\curves_S" --b "s6=$Out\curves_s6" --b "Ss6=$Out\curves_Ss6" --out "$Out\paired_vsA.json" 2>&1 | Tee-Object "$Out\paired_vsA.log"
  & $Py work4\paired.py --boot $Boot --base "S=$Out\curves_S" --b "Ss6=$Out\curves_Ss6" --out "$Out\paired_vsS.json" 2>&1 | Tee-Object "$Out\paired_vsS.log"
  & $Py work4\paired.py --boot $Boot --base "s6=$Out\curves_s6" --b "Ss6=$Out\curves_Ss6" --out "$Out\paired_vss6.json" 2>&1 | Tee-Object "$Out\paired_vss6.log"
} finally { Pop-Location }
# --- 4. точная оценка в порядке СДАЧИ (пороги по фолду → multipole+filter из ветки v2 → match по семьям) ---
$env:B10 = "$T\b10"; $env:V2_EXPDS = "$T\v2\expds"; $env:FAM_A = "$Out\fam_A"; $env:FAM_S = "$Out\fam_S"
$env:CURV_A = "$Out\curves_A"; $env:CURV_S = "$Out\curves_S"; $env:GT = $GT; $env:CLUSTERS = $CL
& $Py "$T\bot7\geo_exact.py" "$Out\geo_exact.pkl" 2>&1 | Tee-Object "$Out\geo_exact.log"
& $Py "$T\bot7\geo_stats.py" "$Out\geo_exact.pkl" "$Out\geo_stats.json" 2>&1 | Tee-Object "$Out\geo_stats.log"
"замер: {0:N0} c" -f ((Get-Date) - $t0).TotalSeconds | Tee-Object -Append "$Out\time.txt"
# --- 5. (необязательно) main.py кандидата на одном участке: время и пиковая память ---
if ($MainRegion -and $Zip) {
  $Z = "$Out\zip_" + [IO.Path]::GetFileNameWithoutExtension($Zip); if (-not (Test-Path "$Z\inference\main.py")) { Expand-Archive $Zip -DestinationPath $Z }
  $o = "$Out\main_out\result.geojson"; New-Item -ItemType Directory -Force (Split-Path $o) | Out-Null
  $t1 = Get-Date
  $p = Start-Process -FilePath $Py -ArgumentList @("$Z\inference\main.py", "`"$MainRegion`"", "`"$o`"") -WorkingDirectory $Z -PassThru -NoNewWindow `
       -RedirectStandardOutput "$Out\main.log" -RedirectStandardError "$Out\main.err"
  $peak = 0; while (-not $p.HasExited) { try { $p.Refresh(); $peak = [Math]::Max($peak, $p.PeakWorkingSet64) } catch {}; Start-Sleep -Milliseconds 500 }
  "main.py: {0:N1} c, пик WS {1:N0} МБ, код {2}" -f ((Get-Date) - $t1).TotalSeconds, ($peak/1MB), $p.ExitCode | Tee-Object -Append "$Out\time.txt"
}
# Проверка исправления: в changed_families.txt для каждого варианта должно быть 3 единицы (Гнёздово, Семибратнее, Судьбищи или их кластеры), и Δ к A ≠ 0.
# Проверки: в paired_vsA.log строка k10 — A_keep 0.333237, A_drop 0.314615 (bot-10 REPORT4); иначе stack_base не тот, числа не сравнивать.
# Коробка (для сверки): bot-10 порядок — S +0.011935/+0.010980, s6 +0.015057/+0.006703, Ss6 ≈ +0.01347/+0.011041 (keep/drop);
# порядок сдачи (bot-7 work5/exact/geo_stats.log): S +0.012634/+0.010949, s6 +0.012285/+0.003768, Ss6 +0.013938/+0.011009.
