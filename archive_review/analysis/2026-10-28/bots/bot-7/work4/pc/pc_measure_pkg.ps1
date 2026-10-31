# bot-7, 28.10.2026: честный перемер пакета-кандидата НА ПК. Читает ТОЛЬКО C:. Ничего не посылает (ни LB, ни Kaggle, ни push).
# Пакет: stack6 separate + курганы 0,5 м + crop verify (bot-8) [+ kurgan_twins_only при -Twins]; реранк архитектуры
# прибавляется по классу из строгого вложенного замера bot-10 (классы не пересекаются, см. REPORT3 §5).
#   без twins:  powershell -ExecutionPolicy Bypass -File C:\expds3c\bot7\pc_measure_pkg.ps1
#   с twins:    powershell -ExecutionPolicy Bypass -File C:\expds3c\bot7\pc_measure_pkg.ps1 -Twins
param(
  [switch]$Twins,
  [string]$Repo  = 'C:\Sloi\expds-koz3_v0.2.0\expds-koz3',      # ДОПУЩЕНИЕ: путь репо на C: (поправить при отличии)
  [string]$Py    = 'C:\Sloi\expds-koz3_v0.2.0\expds-koz3\.venv\Scripts\python.exe',
  [string]$In    = 'C:\expds3c\in',                             # входы, скопированные с F: и с коробки (список ниже)
  [string]$Out   = 'C:\expds3c\pkg',
  [string]$Bot7  = 'C:\expds3c\bot7',                           # candidate_measure.mbox, family_clusters.json
  [string]$Crop  = 'stack_verify_sp.geojson'                    # ДОПУЩЕНИЕ: имя стека crop verify (docstring scripts/crop_verify.py)
)
$ErrorActionPreference = 'Stop'
# --- 0. только C: ---
foreach ($p in @($Repo,$Py,$In,$Out,$Bot7)) { if (-not $p.ToUpper().StartsWith('C:\')) { throw "не C:: $p" } }
$need = @("$In\$Crop", "$In\stack_base.geojson", "$In\gt_train_v2.geojson", "$Bot7\family_clusters.json", "$Bot7\candidate_measure.mbox")
foreach ($f in $need) { if (-not (Test-Path $f)) { throw "нет файла: $f (см. REPORT3 §6: что скопировать на C:)" } }
$env:OMP_NUM_THREADS = '2'; $env:MKL_NUM_THREADS = '2'; $env:PYTHONDONTWRITEBYTECODE = '1'
$Wt = 'C:\expds3c\wt_bot7'
# --- 1. код: отдельный worktree на C: = база 24bd916 + ветка замера bot7-candidate-measure (mbox) ---
if (-not (Test-Path $Wt)) {
  git -C $Repo cat-file -e 24bd916^{commit}
  git -C $Repo worktree add --detach $Wt 24bd916
  git -C $Wt -c user.name=bot-7 -c user.email=bot7@local am "$Bot7\candidate_measure.mbox"
}
$env:EXPDS_METRIC = "$Wt\materials\compute_metrics.py"
New-Item -ItemType Directory -Force $Out | Out-Null
$t0 = Get-Date
# --- 2. посемейные стеки (один раз на оба варианта) ---
if (-not (Test-Path "$Out\fam_A\_counts.json"))    { & $Py "$Wt\scripts\split_stack_by_family.py" --stack "$In\stack_base.geojson" --out "$Out\fam_A" }
if (-not (Test-Path "$Out\fam_crop\_counts.json")) { & $Py "$Wt\scripts\split_stack_by_family.py" --stack "$In\$Crop" --out "$Out\fam_crop" }
if (-not (Test-Path "$Out\fam_crop_kur\_counts.json")) {
  & $Py "$Wt\scripts\micro_fam.py" --fam "$Out\fam_crop" --out "$Out\fam_crop_kur" --radius-map kurgany_povrezhdennye=0.5,kurgany_tselye=0.5 }
# --- 3. пакет: stack6 separate [+ twins] ---
$tag = if ($Twins) { 'pkgT' } else { 'pkg' }
$tw  = if ($Twins) { @('--twins') } else { @() }
& $Py "$Wt\scripts\pkg_fam.py" --fam "$Out\fam_crop_kur" --out "$Out\fam_$tag" --radius-map '' @tw | Tee-Object "$Out\pkg_fam_$tag.log"
# --- 4. кривые и честный бутстрэп (unseen keep), A = база без crop ---
foreach ($d in @('A','crop')) { if (-not (Test-Path "$Out\curves_$d")) {
  & $Py "$Wt\scripts\family_curves.py" --fam "$Out\fam_$d" --gt "$In\gt_train_v2.geojson" --clusters "$Bot7\family_clusters.json" --out "$Out\curves_$d" } }
& $Py "$Wt\scripts\family_curves.py" --fam "$Out\fam_$tag" --gt "$In\gt_train_v2.geojson" --clusters "$Bot7\family_clusters.json" --out "$Out\curves_$tag"
& $Py "$Wt\scripts\honest_bootstrap.py" --a "$Out\curves_$tag" --b "base=$Out\curves_A" --b "crop=$Out\curves_crop" `
   --boot 2000 --lb 2000 --seeds 5 --unseen keep --out "$Out\hb_$tag.json"
"{0}: {1:N0} c" -f $tag, ((Get-Date) - $t0).TotalSeconds | Tee-Object -Append "$Out\time_$tag.txt"
# Внимание: имя «A» занято под --a (honest_bootstrap.py:164), база идёт как «base». Проверка: base keep должен быть 0,333237 (bot-10 REPORT4); иначе стек базы не тот — числа не сравнивать.
# Итог пакета = Δ(hb_$tag vs A) + ΔR реранка архитектуры +0,009536 (bot-10 строгий вложенный, только arkhitektury).
