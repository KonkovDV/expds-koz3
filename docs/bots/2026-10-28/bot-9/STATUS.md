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
