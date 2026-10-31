# Для bot 9: D-FINE-L / RT-DETRv2 на Kaggle (bot-5, 28.10.2026)

**Только честный замер. Ничего не идёт в сдачу до письма организаторов** (лицензия весов HF; AI_WORK_PLAN 2.15).

## Где код
Ветка `bot5-dfine-kaggle`, коммит `3027dc7` (не запушен; поверх 0256251 — пин версий и проверка meta) в `/workspace/koz3_shared/bot-5/repo`:
* `scripts/det_train_hf.py` — тренер/предсказатель (transformers.AutoModelForObjectDetection; код конкурентов не использован).
  Модели: `dfine-l` (ustc-community/dfine-large-coco), `dfine-s`, `rtdetrv2-r50` (PekingU/rtdetr_v2_r50vd).
  Флаги: `--data --out --model --size --bs --iters --fp16 --tta-flip --smoke --dry-run --num-dn --strong-aug --pred-data DIR --pred-split valid --load MODEL_DIR`.
* `kaggle/dfine_folds.sh <DET_ROOT> <OUT_ROOT> [MODEL] [ITERS]` — 2 полосы (GPU0 f0 f2 f4; GPU1 f1 f3); на фолд: обучение
  `--iters 8000 --bs 8 --size 640 --fp16 --tta-flip`, затем `--load OUT/fk/model --pred-data DET/grid_fk --pred-split valid` → `OUT/fk/grid/preds_valid.json`.
* `kaggle/dfine_kaggle.ipynb` — pip → пути REPO/DET/OUT → smoke → полосы → сводка timing → `dfine_preds.tgz`.
* `kaggle/README_KAGGLE.md` — полный рецепт (шаги 0–4, команды дословно).

**Версии — точно:** `pip install -r kaggle/requirements_dfine.txt` → `transformers==5.21.0`, `timm==1.0.30` (проверены на боксе: dry-run D-FINE-L ок, 848 МБ, `tmp/dfl_dry_pin.log`). Не `>=`: в transformers 5.x `from_pretrained` подменяет `tie_weights` на весь процесс, и при загрузке двух моделей в параллельных потоках `class_embed`/`bbox_embed` остаются на meta. После загрузки, до `.to(device)`, `assert_no_meta` падает с понятной ошибкой. Модели не грузить в потоках; фолды — отдельные процессы.

## Порядок
0. (машина с данными, CPU) `expds.tiles` → `cache/spor_det05` и `cache/spor_det05_grid --grid-overlap 0.25`; `coco_from_cache.py` → `det/f0..4`, `det/grid_f0..4` (README шаг 0). `det/` → Dataset `koz3-spor-det05`, репо → Dataset `expds-koz3-bot5`.
1. **Smoke на T4 (первый шаг):** `python scripts/det_train_hf.py --data det/f0 --out runs/smoke --model dfine-l --size 640 --fp16 --smoke` → `runs/smoke/timing.json` (s_per_iter, cuda_peak_MB). Бюджет полосы GPU0 ≈ s/ит × ITERS × 3 + сетка; не влезает в 12 ч → `ITERS=6000` или фолды по сессиям.
2. `bash kaggle/dfine_folds.sh det runs/dfine dfine-l 8000` (потом то же с `rtdetrv2-r50` → `runs/rtv2`). При NaN/коллапсе — `--num-dn 0`.
3. Скачать `dfine_preds.tgz` → машина с данными: README шаг 4 (`coco_preds_to_stack.py` по фолдам `--coco det/grid_f$k/valid/_annotations.coco.json` → слияние → `scoreboard.py <флаги A> --add work/sp_dfine_oof.geojson:sp_dfine --honest 10` и `--replace sp_focal=...`).
4. Приёмка П1–П6 (`/workspace/koz3_shared/EXPERIMENTS.md:13-20`). Пороги — только honest-10/LOFO, не на той же семье.

## Проверено на боксе (CPU, 2 потока; не оценка для T4)
| проверка | итог | пик RSS | файл |
|---|---|---|---|
| smoke D-FINE-L, 256 px, bs 2 | 12,19 с/ит, всего 139,3 с, 400 рамок / 4 окна | 1577 МБ | bot-5/tmp/dfl_smoke/timing.json, tmp/dfl_smoke.log |
| dry-run D-FINE-L | ок | 841 МБ | tmp/dfl_dry |
| dry-run RT-DETRv2 (42,7 M пар.) | forward 4,6 с, loss 277,9 | 867 МБ | tmp/rtv2_dry.log |
| dfine-s `--iters 1` + сохранение, затем `--load` → 400 рамок | ок | 774 / 592 МБ | tmp/dfs_save.log, tmp/dfs_load.log |
| формат: smoke-предсказания → coco_preds_to_stack | 68 объектов sp_dfine | — | tmp/dfl_smoke/stack_sp_dfine.geojson |
Ограничение памяти на боксе: `ulimit -v 4000000` + сторож RSS 2000 МБ (`work/run_smoke.sh`); при `ulimit -v 2500000` torch не стартует (резерв вирт. памяти).
