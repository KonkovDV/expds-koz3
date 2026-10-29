# STATUS bot-9 (Kaggle-очередь)
- 28.10: прочитаны COMMON.md, expds/train.py, expds/centers.py, scripts/kaggle_dual_t4_orchestrator.py, cloud_gpu_farm.py, kaggle_collector_daemon.py, /workspace/kaggle_out/*.
- Метаданные чекпойнтов без torch: `python3 ckpt_meta.py <pt>` (ckpt_meta.py в этой папке).
- Фолды пересчитаны по index.csv: configs/folds_li_tin_59_t128.json.
- Диагноз edge/centers: EMA 0.999 при ~1.3–1.7k шагах + порог пика 0.5 + радиус совпадения 1 м + FPN stride 4 + edge-ignore; чекпойнт эпохи 1 (см. REPORT.md).
- Дальше: патч train.py/centers.py в ветке bot9-kaggle-queue, 3 ноутбука, REPORT.md.
- Патч закоммичен (b26abea), bot9.diff; ноутбуки notebooks/0{1,2,3}_*.ipynb, configs/*.json; проверки: py_compile, dry-run 32/32, ast ячеек, мок-тесты guard/run_queue/apply.
- REPORT.md написан. Сообщений никому не отправлял, Kaggle/ПК/GPU не трогал.
- 28.10 20:09–20:26 МСК, задание 2 (от бота «1»):
  - 01 v2 собран: kaggle/01_kurgan_subclass/{koz3-kurgan-subclass.ipynb,kernel-metadata.json}, configs/01_kurgan_subclass.json; тесты tests/k01 (run_local.log, test_hf.log) OK. Не пушил.
  - s5: COMPLETE (20:16), выход скачан в /workspace/kaggle_out/koz3-gpu-s5 (GPU OK, 12/12 rc=0); stage_s5_for_nn_oof.py + команды OOF для ПК в REPORT2 §2.
  - s4: RUNNING на 20:16/20:21/20:23/20:24/20:25; не скачан. Готовы eval_s4_oof.py и repo/scripts/s4_rerank.py (коммит 194a691, bot9.diff обновлён).
  - REPORT2.md написан. Сообщений не отправлял, на Kaggle ничего не запускал.
  - 20:28 рутина: s5 COMPLETE (уже обработан), s4 RUNNING — без изменений, ничего не отправлял.
  - 20:40 рутина: s5 COMPLETE (уже обработан), s4 RUNNING — без изменений, ничего не отправлял.
- 20:48 бот «1» (рутина): s4 COMPLETE, выход скачивается в /workspace/kaggle_out/koz3-gpu-s4; запущено ядро dmitrijkv/koz3-kurgan-subclass v1 (01, датасет crops готов). Не пушить повторно; скачать выход по завершении.
- 20:49 s4 скачан, eval_s4_oof rc=0 -> s4_eval.json
- 20:58 Task3: REPORT2 §6 (S3) записан; VERIFIER.md — заглушка (отложено). Начато: matanerdy_repro.
- 21:05 рутина: s4 COMPLETE обработан — REPORT2 §3/§5 дополнены цифрами из s4_eval.json; оба ядра обработаны, REPORT2 финален; rerank/П1–П5 только на ПК. Рутина удаляется.

## 28.10.2026 ~21:20 MSK — steering 21:05/21:06/21:08
- matanerdy_repro: отменено (REPRO.md, 2 строки). kurgan-subclass: снят, фикс smp не доводился. Находка: оркестратор кладёт smp в бандл, только если на момент сборки он был в /tmp/expds-venv/…/python3.12, поэтому в tar его, скорее всего, нет; локальная копия /tmp/bot9_hf — пустой мок.
- nolidar_verifier: ГОТОВ К ПУШУ, не запушен → NOLIDAR_VERIFIER.md (pip в чистом venv OK, сухой прогон на 3 участках из S3 OK).
- nolidar_seg: следующий шаг (после resume).
- 21:25 MSK: повторный CPU-сухой прогон nolidar_verifier (версия бота 1) — OK, maxrss 1898 МБ (dryrun2/dryrun.log). nolidar_seg — ОТЛОЖЕН. Дальше: dfine_spor.
- 21:38 MSK: dfine_spor — ядра smoke/full собраны (kaggle/dfine_spor/{smoke,full}), НЕ запушены. pip в чистом venv OK (pip_clean.log). CPU-мини-прогон end-to-end OK: 446 детекций, maxrss 1269 МБ (dryrun/dryrun.log). Исправлен boundless-read WarpedVRT. Отчёт DFINE.md. Тренера бота 5 пока нет — используется встроенный, хук dfine_train_bot5.py. ConvNeXt-T / RELIEF / nolidar_seg не начаты.

## 28.10.2026 ~21:45 MSK — nolidar_verifier v1: COMPLETE (Kaggle)
- Запуск ботом 1 около 21:20 MSK (с его слов); работа заняла 555 с на 2×T4 (`summary_verifier.json`).
- Выход: `/workspace/kgl_out/nolidar_v/` (oof_verifier_nolidar.csv, summary_verifier.json, лог, weights_{kurgan,gor}_fold0-4.pt).
- Разбор: NOLIDAR_VERIFIER.md, раздел «Итог Kaggle v1»; скрипты и логи — `nolidar_v1_analysis/`.
- Итог kurgan: верификатор AUC 0,586 против conf 0,725 (поле «auc_stack_conf» — это conf в одиночку, а не стек); лучший стек — 0,720. Пользы нет, боту 10 не отдавать.
- Итог gor: 0,698 против 0,619 при 18 позитивах. Только как дешёвая проверка с низким приоритетом.
- Разрыв val/test: выбор эпохи по максимуму на одной семье (Тыва) против пула из многих семей с разной калибровкой; провал на семье «Курганы» (0,520). Утечки нет.
- Дальше (v2, не запускать без п. 1–3): val из нескольких семей или фиксированные эпохи; остаток над conf с нормировкой по участку; просмотр кропов Гнёздово и Курганы_34/29.
- Ничего не пушил, на Kaggle не запускал, сообщений не отправлял.

## 28.10.2026 ~22:45 MSK — ConvNeXt-T U-Net по лидару: СОБРАНО, НЕ ЗАПУШЕНО
- 12 ядер в kaggle/convnext_lidar/ (prep CPU, smoke, 10 полных 2×T4); голова центров bot-2 centerhead @ 788202a (вендор по git, read-only).
- Проверки: build OK, test_head rc=0, сухие прогоны centerhead/heat exit 0, prep→smoke-из-prep OK, validate_oof_v2 без ошибок, check_maps 428/428 и 1669/1669, grep ключей = 0.
- OOF-карты npz + region_fold.json для реранкера бота 2; HEAD_INTERFACE.md с «Ответ боту 2 по п.5».
- Отчёт: CONVNEXT.md. Ничего не пушил, сообщений не отправлял.

## 28.10.2026 ~22:38 MSK — D-FINE smoke v2 разобран; full разбит на 2 ядра; НЕ ЗАПУШЕНО
- Smoke v2 (Kaggle): все шаги OK, ошибок фолдов нет, loss конечны. T4 0,92 с/ит (`smoke_v2_out/koz3-dfine-spor-smoke.log`). Рост loss — шум: CPU-тренд 32 ит., фиксированный батч 108,6 → 60,6 (`fix/smoke_v2_check/train256_f3.log`).
- Баг чипов найден и исправлен: терялись объекты на маленьких растрах и у краёв (было 38 объектных чипов на 89 GT, стало 89 на 89) + недетерминированный фон из-за hash(). Добавлены peak VRAM в отчёте и пропуск инференса после дедлайна.
- Бюджет: full = фолды 4,2,1, 6000 ит., потолок 5 ч, прогноз ≈ 3,7 ч. full_rest = фолды 0,3, ≈ 1,9 ч (новое ядро `kaggle/dfine_spor/full_rest`). **28.10: ГОТОВ, пуш сразу после COMPLETE full, до cnx-li-smoke (решение пользователя).**
- Проверки: build + py_compile ×3, CPU-дрираны smoke/full rc=0 (RSS ≤ 1,52 ГБ), grep ключей = 0. Подробности и команды пуша — DFINE.md, разделы «Smoke v2», «Бюджет», «Пуш».
- Ничего не пушил, на Kaggle не запускал, сообщений не отправлял.
- 22:39 МСК рутина бота 1: запушен koz3-dfine-spor-full (фолды 4,2,1; GPU простаивал с ~22:07 smoke v2). Ключи: grep AWS_* = 0 (совпал только эндпоинт S3EP, не секрет).

## 28.10.2026 ~22:55 MSK — ConvNeXt по лидару для TPU (scratch c3sc/c5sc): СОБРАНО, НЕ ЗАПУШЕНО
- 4 ядра в kaggle/convnext_lidar_tpu/ (smoke, sc, c3sc, c5sc), TpuV5E8 (v3-8 выведен), build.py из GPU src без изменений GPU-папок.
- Один процесс на чип (не xmp.spawn); bf16 autocast; XAdamW с тензорным lr; лосс centerhead без синхронизаций; статические формы.
- Проверки на боксе: build OK, тесты частей CPU/XLA OK, dry CPU (главный + 1 воркер) rc 0, dry XLA-CPU rc 0, 0 перекомпиляций, 0 host-sync в шаге, grep ключей 0.
- Одновременность TPU+GPU официально не подтверждена — проверить smoke. Критерий: TPU >1.5× медленнее T4 на сэмпл → закрыть.
- Отчёт: CONVNEXT_TPU.md. Ничего не пушил, сообщений не отправлял.

## 28.10.2026 ~23:59 MSK — контракт весов для bot-8 + бенч bot-8 в smoke (по запросу бота 8 через бота 1)
- Документ: CNX_CONTRACT_FOR_BOT8.md.
  - Каналы несовместимы: 3/5 против 9/12; другая ЦМР; сдвиг 1 px; Δ входа до 5,0 (parity_bot8/compare.json).
  - Рекомендация: у bot-8 in_kind cnx3/cnx5 на нашем вендоре.
- Код:
  - src/convnext_main.py: «Шаг 9б» weights_cnx_*.pt и «Шаг 11» бенч (только smoke).
  - build.py: BENCH_B64 из bot8-cpu-a-clean @95bc0a6.
  - Новое: src/cnx_contract.py, tests/test_contract.py, tests/test_bench_wrapper.py.
- Пересборка: GPU 12 + TPU 4 ядер, RELIEF_SHA тот же, metadata не изменились, py_compile 26/26.
  - sha256: smoke 508cb3d2db…, c5sc-f01 f56a62c4…, c3sc-f01 5d45d975…, TPU smoke 57752e45…, TPU sc 12828206…
- Команда пуша smoke не изменилась. Очередь GPU: D-FINE full → full_rest (dfine_spor не трогал) → koz3-cnx-li-smoke.
- Бэкап до изменений: backup/cnx_contract_2340/. Ничего не пушил, сообщений не отправлял.
- 28.10 ~23:59 MSK, для бота 8 (ядра не трогал): CNX_CONTRACT_FOR_BOT8.md §6 — точка входа `for_bot8/cnx_prep.py:cnx_input` (паритет с prep-ядром max|Δ| 0,0 на Осечки/Хога/Эсбю, в т.ч. в venv bot-8 с их expds_core); у bot-8 не хватает только rvt-py (колесо for_bot8/wheels/, --no-deps); smp 0.5.0 = 0.6.0 по выходу; bf16 на CPU в 10–30× хуже fp16 → на 4090 оставить fp16.
- 29.10 ≈02:30 МСК — разбор koz3-dfine-spor-full (бот 9): обучение 6000/6000 ×3, 11 044 с, без дедлайна; OOF 203 131 Polygon EPSG:3857, 168 невалидных тонких эллипсов, поле `group` вместо `threshold_group`. Метрика (бот-10 конвейер): все семьи A 0,314615 → +D-FINE 0,331490 (Δ +0,0169, P 0,934, LOFO +0,0200, 5/5, П1–П4 ✓); на семьях ф. 4/2/1 Δ +0,0438; на sp_or D-FINE один 0,2756 > стек 0,1662. Исправлен инвентарь GT: 1606 (ф.4/2/1 — 1357). Подробно — DFINE.md, `kaggle/dfine_spor/full_out/`.
- 29.10 ~02:32 МСК — cnx перед GPU (бот 9): в `convnext_main.py` threshold_group="li_cnx" в OOF и бюджет по медиане с/ит ит. 101–150; тесты rc 0, сухой прогон exit 0; пересобраны 12 GPU-ядер + только tpu-sc (tpu-smoke/c3sc/c5sc не тронуты). RELIEF_SHA, вендор, metadata, команды пуша без изменений. Подробно — CONVNEXT.md, sha — `kaggle/convnext_lidar/logs/sha256_pre_gpu_0235.txt`. Бэкап — backup/cnx_pre_gpu_0235/.
- 29.10 ~04:50 МСК — full_rest D-FINE (фолды 0,3) разобран (бот 9): 6000/6000 ×2, 6679 с, n_det 2516/37382 (f0 мало из-за 104 окон Гнёздово, не баг); OOF 5 фолдов `full_rest_out/oof_dfine_5f.geojson` 243 029 объектов, threshold_group=sp_dfine, 443 починены buffer(0). Метрика 5 фолдов: A 0,314615 → C 0,334414 (Δ +0,0198, P 0,985, LOFO +0,0253, 5/5, П1–П5 ✓), E +0,0199, R +0,0198 (R−C ≈0); фолды 0,3: C +0,0093 (✗), R +0,0229 (✓). sp_or: D-FINE один 0,2666 > стек 0,1532. Смоук cnx — наша сборка (sha 95d72749…), OOF li_cnx. Подробно — DFINE.md.
- 29.10 ~06:25 МСК — koz3-cnx-li-c3sc-f01 разобран (бот 9): 12000/12000 ×2, бюджет не урезан (медиана 0,322/0,343 с/ит), 4244 с, VRAM не логируется. li-only f1 0,4703 / f0 0,2168 (без порога, ≥0,05). 9604 в f0 — в основном хвост <0,3 (≥0,3: 1003 курганов на 837 GT), но калибровка разная по участкам (лучший порог povr: Гнёздово 0,33, Тимерево 0,06). Стек + li_cnx (2 семьи): Δ −0,0105 (Xx без дублей классов −0,0081), П1–П4 ✗. Причина — порог по одной семье (Тимерево 0,05 → 5142 FP в Гнёздово); ждать все фолды. Подробно — CONVNEXT.md, `kaggle/convnext_lidar/c3sc_f01_out/metric/`.
- 29.10 ~06:37 МСК — convnext_main.py 6f0087a8…: лог/отчёт VRAM + data_wait по полосам, пороги (0.3 и лучший по классам) рядом с li-only, CFG после CIRCLE_R; тесты + сухой прогон OK, 12 ядер пересобраны (c5sc-f01 в 06:30), metadata/команды пуша без изменений, c3-f4 возвращён из hold/ (03c1366c…) — CONVNEXT.md (бот 9)
