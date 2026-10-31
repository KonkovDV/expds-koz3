# bot-9 · Очередь Kaggle (2×T4): диагноз edge/centers, ноутбуки, план GPU-часов
28.10.2026 (МСК). Ничего не запускалось на GPU/Kaggle/ПК: только чтение, py_compile, argparse `--dry-run`, синтетические CPU-тесты.

## Вывод (3 строки)
1. «Edge»-модели — не края, а карта центров (`target=centers`, sigma 0, radius 4 м, edge-ignore) в неподходящем рецепте: FPN r34 + EMA 0.999 без разгона, F1 считается на пороге пика 0.5 при радиусе 1 м, и отбор чекпойнта сохраняет эпоху 1–2 → center_f1 ≤ 0.045 при pixel F1 маски 0.35–0.64.
2. Наибольший ожидаемый прирост — классификатор подкласса кургана (повреждённый/целый): идеальный разбор «близнецов» даёт ≈ +0.016 к баллу, при исправлении непарных путаниц — до +0.03…0.05; карта центров и ансамбль семян — низкий/отрицательный приор.
3. Подготовлены патч (ветка `bot9-kaggle-queue`, `bot9.diff`), 3 ноутбука + конфиги с жёсткой GPU-защитой и leave-family-out; вся очередь ≈ 3.2 ч сессии 2×T4 (≈ 6.4 T4-ч), с резервом ≈ 4.9 ч (≈ 9.8 T4-ч) — < 17 % доступного до 11.11.2026 23:59 МСК.

## 1. Находки (с путями)
### 1.1 Что такое «edge» и почему center_f1 = 0.045
| # | Факт | Ссылка |
|---|---|---|
| a | Чекпойнты `li_r34_fpn_edge_59_fold*` — `target=centers`, `center_sigma_m 0.0`, `center_radius_m 4.0`, `min_dist_m 12.0`, `center_edge_ignore true`, FPN resnet34, `ema 0.999` | `python3 ckpt_meta.py <pt>` (этот каталог) по `/workspace/kaggle_out/koz3-gpu-s2/*.pt`, `koz3-gpu-s3/*.pt` |
| b | Сданный рецепт центров другой: Unet r18, без EMA, без edge-ignore, `--center-sigma 0 --center-radius 2.0 --center-min-dist 4.0` | repo/docs/AI_WORK_PLAN_2026-10-25.md:297; edge-ignore по умолчанию выкл. — docs/PLAN.md:59 |
| c | EMA 0.999 без разгона: шагов за 30 эпох ≈ 1290–1650 (43–57 шагов/эпоху), 0.999^1290 = 0.275, 0.999^1650 = 0.19 — до четверти веса — копия модели шага 1, включая BN-буферы (`use_buffers=True`) | repo/expds/train.py:502–509 (HEAD 24bd916) |
| d | Следствие в логах: эпоха 1 train 1094.469 / **val 10876.051**, эпоха 2 val 2274.519; эпоха 30 val 4.5158, center_f1 все 0.0 | /workspace/kaggle_out/koz3-gpu-s2/wave1_slot0_step0.log:4,5,33 |
| e | Лучший фолд: эпоха 30 center_f1 povr 0.014, tselye 0.076 (среднее best 0.0453) | koz3-gpu-s2/wave1_slot1_step1.log:33; repo/notes/kaggle_gpu_s1_benchmark_2026-10-28.json (best F1 edge 0, 0, 0.0024, 0.0195, 0.0453) |
| f | Та же EMA и на маске r34 FPN: эпоха 1 val 2.142, F1 povr 0.122 против 1.3156 / 0.301 у r18 Unet без EMA; к эпохе 30 r34 догоняет (0.419 / 0.638) — EMA тормозит, но маска выживает за счёт гладкой цели | koz3-gpu-s1/wave2_slot0_step0.log:2,31; wave1_slot0_step0.log:2,31 |
| g | F1 центров на проверке строже метрики: пик > 0.5 при prior головы 0.01 (инференс использует порог 0.05), радиус совпадения `max(2, round(sigma_px))` = 2 px = 1 м при sigma 0 (метрика — 2 м), FP с весом 0.5 | train.py:462, 648–668; expds/centers.py:179–202; materials/compute_metrics.py:696 |
| h | FPN выдаёт карту stride 4 с билинейным ×4 — однопиксельный шип (sigma 0) размывается, пик > 0.5 почти недостижим | smp FPN; train.py (arch FPN), centers.py:23–76 (`by_radius`, spike) |
| i | Отбор чекпойнта: `better = f1 > best+1e-6 or (равны и vl < best_loss)`; при F1 = 0 почти на всех эпохах в файле остаётся эпоха 1–2 с F1 0.00x | train.py:674–698; метаданные: s2 fold4 **ep1 val 1050.67**, s3 fold1 **ep1 val 15517.45**, s3 fold2 **ep2 val 1086.53** (ckpt_meta.py) |
| j | Эпоха центров в 3–7 раз медленнее маски (12.4–27.2 с против 3.6–4.3 с): при spike+flips цель считается на CPU (scipy) в каждом батче, предрасчёт отключён условием `not center_spike or not flips` | train.py:529–542; логи выше; GpuTiles отражает `y_pre` вместе с `x` — train.py:332–334 |

**Причина расхождения edge/centers:** не CRS, не порядок каналов и не нормализация — они совпадают (x/255, (x−0.5)/0.25, каналы `lrm_s/lrm_l/slope`, res 0.5, tile 128 у обоих; meta.json бандла и метаданные чекпойнтов). Расхождение — рецепт и учёт: (c) EMA 0.999 без разгона + (h) FPN stride 4 для 1-px шипа + (g) порог 0.5 и радиус 1 м в F1 + (i) отбор эпохи 1–2. Edge-ignore (a) дополнительно выбрасывает обрезанные краем насыпи.

### 1.2 Очередь и честность проверки
| # | Факт | Ссылка |
|---|---|---|
| k | `splits_val_map` не определён нигде → `plan`/`stage` в HEAD падают с NameError | scripts/kaggle_dual_t4_orchestrator.py:170–281 (HEAD) — исправлено |
| l | Финальные модели шли с `--val-families auto`: такой семьи нет → train берёт «10 % хвоста» индекса (часть семьи в проверке, часть в обучении; на финал не все данные) | оркестратор :205, :281; train.py:448–453 |
| m | Фолды по семьям честные: в кэше фолда k нет тестовых семей; семья проверки всегда Муравлево (240 тайлов) или Боровичи (383) | configs/folds_li_tin_59_t128.json (пересчёт по /workspace/kaggle_out/bundle_index/li_tin_59_t128/index.csv тем же правилом, что оркестратор :127–152; сходится с логами) |
| n | Выполненные команды были без `--require-gpu` (`<auto_val_family>`) — логи от версии до f5b3b6d | notes/kaggle_gpu_s1_benchmark_2026-10-28.json, поля cmd |
| o | Второе одновременное ядро на том же аккаунте молча получает CPU (так упал li-tin, exit 1) | оркестратор :637–655; /workspace/kaggle_out/done.log |
| p | Guard train.py переназначал `cuda:N` по модулю числа GPU с одним warning | train.py:475–490 (HEAD) — теперь ошибка |
| q | Коллектор копирует .pt плоско в models_59/: s2 и s3 с одинаковыми именами затирают друг друга, `fold/li.pt` — последний скачанный | scripts/kaggle_collector_daemon.py:32–45 — исправлено |
| r | `crop_verify.fit` молча откатывается на CPU (`cuda if available else cpu`) | scripts/crop_verify.py:146–152 — в ноутбуке 01 своя функция с проверкой устройства |

## 2. Ожидаемый прирост (обоснование через веса метрики)
Веса: povr 30, tselye 20, городища 15, архитектура 10, object_poly 10, фортификации 5, finds 10, нормировка по присутствующим классам; precision = tp/(tp+0.5·fp) (materials/compute_metrics.py:83–91, 696, 753–757). База (честная, лидар+стек): povr TP 1105 / FP 2731 / FN 2515 → F1 0.36286; tselye 711 / 1172 / 866 → 0.49478 (repo/notes/honest_base_2026-10-25.json; F1 воспроизводится формулой). 489 FP povr и 591 FP tselye — настоящие курганы с другим подклассом (docs/REASONING_AND_DISCOVERY_LOG_2026-10-28.md:80); лидар ставит курган в оба подкласса в 0 м (docs/TEAM_REPORTS_2026-10-28.md:51–70).

| Прогон | Расчёт | ΔБалл (оценка) | Приор / риск |
|---|---|---|---|
| **01 подкласс кургана** | идеальный разбор близнецов (FP povr −489, FP tselye −591): F1 0.37804 / 0.55148 → (30·0.01518 + 20·0.0567)/100 = **+0.0159**; если путаницы непарные и переразметка превращает их в TP: 0.52695 / 0.7824 → +0.1068 (верхняя граница); при 30 % / 50 % таких — +0.033 / +0.055 | реалистично **+0.008…+0.03** (допущение: точность подкласса 0.7–0.8, ~½ от идеала) | расчёт — python в этой сессии по honest_base; в журнале резерв +0.045 (REASONING_LOG:115). Двойная метка (−0.005, PLAN.md:127) — другое: там добавляли круги, здесь убираем |
| 02 центры (fix) | прошлый честный опыт карты центров лидара: 0.350588 → 0.351535 (**+0.0009**), 34 новых −0.0021 | ≈ 0…+0.002 | docs/PLAN.md:111; ценность — рабочие 5 OOF-моделей центров вместо мусора и проверка диагноза |
| 03 семена | два семени: **−0.027** | ≤ 0, только как проверка среднего вероятностей | docs/PLAN.md:207, :252 |

## 3. Код / патч
- Клон: `/workspace/koz3_shared/bot-9/repo`, ветка **`bot9-kaggle-queue`**, коммиты `e8d9f84`, `358fd0d`, `b26abea` поверх `24bd916`; push не делался.
- Diff: **`/workspace/koz3_shared/bot-9/bot9.diff`** (507 строк; 5 файлов, +296/−21, `git diff --shortstat 24bd916 HEAD`).
- `expds/train.py`: строгий GPU-guard (`--min-gpus`, отказ при нехватке GPU, запрет переназначения `cuda:N`, проверка устройства всех параметров/буферов модели и первого батча x/y); `--ema-warmup` (decay = min(ema, (1+t)/(10+t))); `init_focal_bias` до копии EMA; `--select-min-epoch`; `--center-match-m` (радиус совпадения в метрах); `--center-peak-thrs` (F1 на нескольких порогах, отбор по лучшему на семье проверки, порог пишется в чекпойнт `val_peak_thr`); `--center-pre-flip` (цель один раз, отражения на GPU; шип может сместиться на 1 px = 0.5 м < 2 м метрики); ошибка, если `--val-families` не нашёл ни одного тайла; `--dry-run`. По умолчанию поведение прежнее.
- `expds/centers.py`: `heatmap_counts(..., thr=0.5)` — порог пика предсказания отдельно от пиков эталона.
- `scripts/kaggle_dual_t4_orchestrator.py`: `splits_val_map` из `compute_fold_splits()`; финал `--all-data` вместо `--val-families auto`. Проверено: план строится, 17 команд, все с `--require-gpu`, семьи проверки {Боровичи, Муравлево}.
- `scripts/kaggle_collector_daemon.py`: копии в `models_59/<ядро>/`, `fold/li.pt` не затирается.
- `scripts/bot9_kurgan_subclass.py` (новый, для ПК): `crops` — окна 64×64 рельефа li вокруг курганов эталона и кандидатов стека (метка — подкласс ближайшего кургана эталона ≤ 2 м); `apply` — переразметка по p_povr ≥ 0.5 и NMS близнецов ≤ 2 м по уверенности. Синтетический CPU-тест `apply`: пара близнецов → 1 круг povr, одиночный — tselye.
- Проверки: `py_compile` всех файлов; `python -m expds.train ... --dry-run` для всех 32 команд конфигов — 32/32 ok; ячейки ноутбуков разобраны `ast`; мок-тест `gpu_guard` (нет CUDA → RuntimeError; 1 GPU при min 2 → RuntimeError) и `run_queue` (команда без `--require-gpu` → RuntimeError). `tests/test_gpu_safeguards.py` не запускался — на машине нет torch; текст ошибки сохранён («Отказ от тихого отката на CPU»).

### Подготовленные файлы
- `notebooks/01_kurgan_subclass.ipynb` — 10 фолдов по семьям (`expds.metric.family_folds(..., site_family, 10)`, assert непересечения семей), потеря 30:20, 2 семени (по одному на GPU), выход `kurgan_subclass_oof.csv` + `out/kurgan_subclass_report.json` (fixed/broken подклассы кандидатов OOF).
- `notebooks/02_centers_fixed.ipynb` — 12 обучений: GPU0 Unet r18 без EMA, GPU1 FPN r34 EMA 0.99 + разгон; оба без edge-ignore, radius 4 м, min-dist 4 м, select-min-epoch 5, match 2 м, пороги 0.05–0.5, pre-flip; smoke 1 эпоха/512 тайлов.
- `notebooks/03_seed_ensemble.ipynb` — li_r18_hard2_59 (маска) семена 1–3 × (5 фолдов + финал) = 18 обучений, по 9 на GPU; smoke.
- `notebooks/bot9_kaggle_common.py` (встраивается в ноутбуки через `%%writefile`): `gpu_guard` (cuda.is_available, device_count ≥ 2, тензор на каждом cuda:i, nvidia-smi, запрет KOZ3_ALLOW_CPU_FALLBACK), проверка патча в бандле, `setup_fold_caches` с проверкой LFO, `run_queue` (проверка `--require-gpu`/`--device cuda:i` до старта, EXPDS_REQUIRE_GPU=1, ошибки потоков всплывают), `check_logs_gpu`.
- `notebooks/build_notebooks.py` — пересборка .ipynb из конфигов.
- `configs/01_kurgan_subclass.json`, `configs/02_centers_fixed.json`, `configs/03_seed_ensemble.json`, `configs/folds_li_tin_59_t128.json`.
- `ckpt_meta.py` — метаданные .pt без torch. `STATUS.md` — заметки.

## 4. Что нужно от ПК / GPU (точные команды; сам не запускал)
1. Пересобрать код в бандле Kaggle (в старом `expds_code.tar.gz` есть smp — менять только файлы из ветки):
   ```
   git -C <repo> fetch /workspace/koz3_shared/bot-9/repo bot9-kaggle-queue && git -C <repo> checkout FETCH_HEAD -- expds/train.py expds/centers.py scripts/
   mkdir code && tar -xzf expds_code.tar.gz -C code && cp <repo>/expds/train.py <repo>/expds/centers.py code/expds/ && cp <repo>/scripts/crop_verify.py <repo>/scripts/bot9_kurgan_subclass.py code/scripts/
   tar -czf expds_code.tar.gz -C code . && kaggle datasets version -p <bundle_dir> -m "bot-9 patch" --dir-mode zip
   ```
   (Если в tar нет `scripts/` — добавить `mkdir code/scripts`.) Ноутбуки сами откажутся работать со старым train.py.
2. Окна для 01 (CPU, на ПК; стек и эталон — те же, что у scoreboard):
   ```
   python scripts/scoreboard.py <флаги раздела 0> --dump-stack F:/expds3/work/stack_base.geojson
   python scripts/bot9_kurgan_subclass.py crops --root F:/2/train-expds-3 --stack F:/expds3/work/stack_base.geojson --gt F:/expds3/work/gt_train_v2.geojson --markup li --res 0.5 --tile 64 --li-tags lrm_s lrm_l slope --out F:/expds3/work/kurgan_subclass_crops.npz
   ```
   → датасет Kaggle `koz3-kurgan-subclass-crops` с этим .npz.
3. Kaggle: импортировать ноутбук, Accelerator **GPU T4 ×2**, подключить бандл (+ датасет окон для 01), Run All; **одно ядро на аккаунт одновременно**.
4. Приёмка на ПК (честно, 10 фолдов):
   ```
   python scripts/bot9_kurgan_subclass.py apply --stack F:/expds3/work/stack_base.geojson --oof kurgan_subclass_oof.csv --thr 0.5 --dedup-m 2.0 --out F:/expds3/work/stack_subclass.geojson
   python scripts/scoreboard.py <флаги раздела 0> --rerank F:/expds3/work/stack_subclass.geojson --honest 10
   ```
   Для 02/03 — распаковать `*_out.tar.gz` в models_59/<прогон>/foldK и прогнать честное табло как для li_focal (docs/PLAN.md:111); для 03 — среднее вероятностей семян, не объединение.

## 5. Очередь и GPU-часы
Основа — реальные времена 30-эпохных обучений на T4 (repo/notes/kaggle_gpu_s1_benchmark_2026-10-28.json): r18 Unet маска 138.8–196.4 с; r34 FPN маска 149.1–228.1 с; r34 FPN центры (edge) 415.6–858.7 с; сессия s1 923.8 с = волны 532.0 + 391.8 с. Время запуска ядра/монтирования в логах не записано → **допущение +0.25 ч на сессию**. Kaggle списывает квоту за время сессии; считаю ли 2×T4 как 1× или 2× — не проверено, поэтому даю обе цифры.

| Приор. | Прогон | Обучений | Расчёт сессии | Сессия 2×T4, ч | T4-ч | Условие запуска |
|---|---|---|---|---|---|---|
| 1 | 01 подкласс кургана | 2 семени × 10 фолдов | нет логов: ~20 тыс. окон 64² (допущение), r18 обрабатывает ~700 тайлов 128²/с (2787 тайлов fold0 за 3.6 с, s1/wave1_slot0_step0.log:31) → ≥2.8 тыс. окон/с → 12 эпох × 10 фолдов ≈ 0.25 ч/GPU | **0.75** (с запасом) | 1.5 | после пересборки бандла и crops.npz |
| 2 | 02 центры fix | 2 × 6 | пессимист.: 6 × 858.7 с = 1.43 ч (если pre-flip не ускорит); оптимист.: 6 × 228.1 с = 0.38 ч | **1.7** (пессимист. + 0.25) | 3.4 | сразу; smoke-ячейка ловит ошибки |
| 3 | 03 семена 1–3 | 2 × 9 | 9 × 196.4 с = 0.49 ч | **0.75** | 1.5 | только если 01/02 дали ≥ 0 на честном табло |
| 4 | резерв: повтор лучшего (01 с `--markup sp_or`/tile 96 или 02 с семенами) | — | как 02 (пессимист.) | 1.7 | 3.4 | по итогам 1–2 |
| | **Итого план** | | | **3.2** (с резервом 4.9) | **6.4** (9.8) | |

Окно: 28.10 → 11.11.2026 23:59 МСК ≈ 2 недели × ~30 GPU-ч/нед ≈ 60 ч (цифра квоты — из постановки; день сброса квоты Kaggle не проверял). План занимает ≤ 4.9 ч сессии (≤ 9.8 T4-ч), т. е. < 17 % даже при двойном списании за 2×T4. Неделя 1 (до ~04.11): 01 + 02 (2.45 ч); неделя 2: 03 и/или резерв (≤ 2.45 ч).
