# PREP PIXGBM clean-room / geo без EN — 01.11 ~10:25 MSK (PLAN_NEXT шаг 7, AUDIT F5/DOC-1)
CPU-only инвентарь. Ничего не обучалось, EN-код (geo_pixgbm.py) не открывался, EN из архивных паков не удалялся, LB/GPU не трогали.

## 0. Вердикт
**READY (CPU) для упаковки и паритета; НЕ BLOCKED по данным.** Все входы для гейта clean-гео против v2 на geo5 на боксе есть,
и сам гейт уже прогнан 30.10 (FREEZE E1_G1). Не хватает одного: **E1_G1 существует только как офлайн-geojson** (скрипты
w4geo/work/*). В inference-коде (origin/main `expds/pix_clean.py`, `pix_org_et.py`) нет его частей. Поэтому clean-пака,
который бы воспроизводил E1_G1 в образе, пока нет. Решение «EN в final или нет» по-прежнему за Еленой.

## 1. Где EN-линия (DOC-1)
| Что | Где |
|---|---|
| `expds/geo_pixgbm.py` (EN, sha 35e4331b…, 849 стр.) | паки: `w4geo/ref/v2bb/`, `out/fr1h-pack/tree`, `out/fr1h-mrp-pack/tree`, `out/verify-spor-pack/tree` (+ zip-ы); копии ревью в `wt/*/analysis/2026-10-29/bots/bot-{4,7}/…`; `wt/w4/expds/`. **В git main нет** (`.gitignore`, docs/PIXGBM_DECISION.md) |
| `models/pixgbm/all/{finds,poly}.txt` (обучены на EN-признаках) | все те же паки + `repo/models/pixgbm`, `expds-koz3/models/pixgbm`, `wt/*/models/pixgbm`; sha в `models/pixgbm/all/MODELS_NOT_IN_GIT.md` |
| Загрузчик в main | `expds/pix_legacy.py` (impl != "clean"; по `check_archive_clean.py` в архив не входит) |
| Конфиг main | `[geophys.pixgbm] enabled = false`, `impl="clean"` → `models/pix_clean` |
| Архив 0.41238 и FR1h/MRP/verify-spor | EN включён: `[geophys.pixgbm] enabled = true`, гео-члены побайтно = архиву |

## 2. Clean-room альтернативы на боксе
| Актив | Путь | Статус |
|---|---|---|
| Код clean pix (в main) | `repo` origin/main f127db8: `expds/pix_clean.py`, `pix_org.py` (ORG1), `pix_org_et.py` (ORG3); `analytic_signal`/`asa` в `expds/geophys.py` (d8e0e76) | есть |
| Clean-модели v2clean (full-train): finds/poly + ORG3 map/prof | `/tmp/v3am/models/pix_clean/` → **скопировано в постоянный** `backups/v3a_min_pix_clean_models_from_tmp/` (SHA256SUMS; /tmp не переживёт ребут) | есть (finds.txt — старая модель, finds −0.15) |
| Скрипты обучения/OOF/гейта | `wt/w4-pix/scripts/`: pix_clean_train/oof/eval, pix_org*_oof, `pix_geo_pr_gate.py`, `pix_modality_gate.py` | есть |
| Растры geo 200–217 | `w4geo/train/` (320 файлов, 1.5G) | есть |
| venv гео (pin lock) | `w4geo/venv` | есть |
| Кэш признаков, фолд-модели geo5 | `w4geo/work/{cache,feat*,models/F{1..5}_s{0..4}}`, `probs_*` | есть |
| Full-train MagGr finds ×6 (ens6) | `w4geo/work/models_{n40r8,n40r6,n40r10,aug,gw1d,gw3d}full/ALL_s0/finds.txt` | есть |
| Кандидат E1_G1 (FROZEN 30.10 23:04) | `w4geo/work/cstar/C_CetO3_Sr8G_noTerFl_L_E1_G1{,_ht}.geojson` (sha c92b25e0/aaba16ef), `orch/bot-W4/FREEZE_E1_G1.md` | есть |
| Рецепт E1_G1 (офлайн) | ens6m mean-map decode fs3 n0.5 (`night/pipe_ens.sh`, `scr_dec.py`), группы Gtype_n300 (`sweep/raster_groups.py`), петли Гнёздово (`gnz/{loops,emit_loops,add_loops}.py`), EtGr top-K25 a≥0.2 (`etfp/mk_E.py`), per-raster norm + soft-NMS (`night/mkM_any.py` q.9 A.25 D1.75 X.6) | есть, **только скрипты** |
| REF v2 на 18 уч. (black-box EN) | `w4geo/ref/out/REF_v2.geojson` (in-sample: EN видела эти участки) | есть |
| Гейт-JSON | `orch/bot-W4/runs/gate*_E1_G1*`, `w4geo/work/cstar/gate_*` | есть |
| Docker + образ pantum/n7q:20261024 | бокс | есть (сегодня гонялся) |

## 3. Числа (что уже известно)
| | geo P / R | MagGr R | протокол |
|---|---|---|---|
| v2 EN на LB | .769 / .656 | — | LB 0.41238 |
| v3a_min clean (MagGr классикой, maps=etgr) на LB | .716 / .311 | — | LB 0.37838 |
| REF_v2 на geo5 | .664 / .785 | .859 | **in-sample** |
| **E1_G1 _ht** | **.693 / .674** | .697 | честный OOF geo5, Δ vs A18 +0.0407…+0.0425, P 1.0 |
| наша full-train in-sample (INS) | .782 / .693 | .720 | in-sample: при любом R доминирует над REF |
Шов: гейт против REF (R .785) структурно несправедлив, потому что REF in-sample. Честный R E1_G1 .674 ≈ LB-R v2 .656. На офлайн-цифрах
clean-путь сопоставим с v2. Перенос OOF → LB не откалиброван: единственная LB-точка clean-линии (v3a_min) — **другой рецепт**,
там MagGr шёл классикой.

## 4. Минимальный план гейта clean-гео против v2 (CPU; без GPU, без LB)
1. **P1 — упаковка E1_G1 в inference (clean).** Новый модуль `expds/pix_e1g1.py` (или флаги в pix_clean) из своих скриптов
   W4 (EN не открывать): mean-map 6 full-моделей → decode fs3/n0.5 → группы Gtype_n300 → петли (только цвет_2) →
   per-raster norm + soft-NMS → EtGr ORG3 top-K25/a0.2 → классический TerFl как есть. Модели: `models/pix_clean/ens6/*`
   (6×finds full) + poly/ORG3 из backups.
   Kill: ОЗУ > +0.35 ГБ на процесс или время гео-фазы > budget_s 600 на 18 уч.
2. **P2 — пороги групп на full-train.** `maggr_pix_{gr,cv}{lo,hi}` и etgr_org на шкале conf G1 (sweep_groups на всех фолдах,
   как `night/insample.py`) → `[thresholds_by_markup.*]`.
3. **P3 — паритет.** Офлайн-композиция full-моделей на 200–217 (как `INS`, но с ens6m) против выхода docker-пака на
   `w4geo/train`. Kill: |Δn| > 1 % или P/R расходятся > .005.
4. **P4 — пак-кандидат** = FR1h-дерево, в котором EN заменён на clean: удалить geo_pixgbm.py + models/pixgbm только в КОПИИ
   (`out/geo-clean-pack/`), `[geophys.pixgbm] impl="clean"`. Затем check_submission (+ правило check_archive_clean)
   и docker-smoke bb_geo/bb_spor. Курганы/не-гео побайтно = FR1h.
5. **P5 — гейт.** Честный OOF уже есть: E1_G1 против A18 PASS, против REF in-sample FAIL по R. Нового офлайн-рычага нет:
   решает LB-проба non-final (Елена). Kill: LB geo R < .60 или final < 0.41238 − 0.005 → в final остаётся EN-вариант
   или вариант «A без pix» по решению Елены.
Оценка: P1–P4 ≈ 4–6 ч CPU-работы одного агента. GPU не нужен.

## 5. BLOCKED / за Еленой
- Честный REF-OOF для EN невозможен без переобучения EN-моделей по фолдам (DOC-1 запрещает открывать/гонять EN-код
  вне black-box) → сравнение «яблоки с яблоками» только через LB.
- Решение для final: (A) без pix, (B) EN с декларацией, (C) clean E1_G1-пак после P1–P5. Плюс «да» на LB-пробу P5.
- Ничего из P1–P4 не запущено в этом проходе: это кодовая работа в новом модуле, ждёт GO от координатора.
