# Для bot 9: RF-DETR-M без COCO-чекпойнта (только DINOv2-S, Apache-2.0) на Kaggle (bot-5, 28.10.2026)

**Только честный замер. Ничего не идёт в сдачу без отдельного решения.** Это запасной путь на случай, если письма организаторов о весах не будет.
Детекторные чекпойнты RF-DETR (COCO/O365) не используются. Энкодер — `facebook/dinov2-small` (Apache-2.0), остальное — с нуля.

## Где код
Ветка `bot5-rfdetr-kaggle`, коммит `b355e2c` (не запушен; перебазирована на `bot5-dfine-kaggle` 3027dc7), `/workspace/koz3_shared/bot-5/repo`:
* `scripts/rfdetr_train.py` — свой тренер/предсказатель. Из `rfdetr==1.11.2` берутся только сборка модели, матчер и критерий.
  Флаги: `--data --out --model rfdetr-m|rfdetr-l --size --iters --bs --accum --fp16 --loss mal|iabce --mal-gamma --cp-prob --cp-max
  --ema --flat --warmup-iters --no-aug-last --dinov2-dir --no-dinov2 --tta-flip --smoke --pred-data DIR --pred-split valid|none --load MODEL_DIR`.
* `kaggle/rfdetr_folds.sh <DET_ROOT> <OUT_ROOT> [MODEL] [ITERS] [DINO_DIR]` — 2 полосы (GPU0 f0 f2 f4; GPU1 f1 f3). На фолд: обучение
  `--size 640 --bs 4 --accum 4 --fp16 --loss mal --cp-prob 0.5 --tta-flip`, затем `--load OUT/fk/model --pred-data DET/grid_fk` → `OUT/fk/grid/preds_valid.json`.
* `kaggle/rfdetr_kaggle.ipynb` — pip (`rfdetr==1.11.2 --no-deps` + supervision, pyDeprecate) → dinov2-small с HF → smoke → замер памяти bs 4 → полосы → `rfdetr_preds.tgz`.
* `kaggle/README_KAGGLE_RFDETR.md` — полный рецепт (шаги 0–4).

**Версии — точно:** `pip install --no-deps rfdetr==1.11.2 && pip install -r kaggle/requirements_rfdetr.txt` → `transformers==5.21.0`, timm 1.0.30, pydantic 2.14.1, supervision 0.31.3, pyDeprecate 0.9.0. HF `from_pretrained` наш скрипт не вызывает: веса DINOv2 читаются из safetensors, а энкодер M/L rfdetr строит конструктором (`dinov2.py:203`). Но классы энкодера — из transformers, поэтому версия закреплена и та же проверка `assert_no_meta` стоит до `.to(device)`. Smoke после правки: 2,16 с/ит, RSS 1795 МБ (`tmp/rf_smoke_pin.log`).

## Порядок
0. Данные — **те же**, что для D-FINE: `det/f0..4`, `det/grid_f0..4` из `coco_from_cache.py --layout rfdetr --folds 5 --fold k` (Dataset `koz3-spor-det05`). Репо → Dataset `expds-koz3-bot5` (ветка `bot5-rfdetr-kaggle`).
1. **Smoke на T4 — первый шаг, только честный замер:** `python scripts/rfdetr_train.py --data det/f0 --out runs/rf_smoke --model rfdetr-m --size 640 --fp16 --smoke`.
   Проверить в логе `dinov2: {'copied': 223, ..., 'missing': 0}` и две строки resampled. Записать `s_per_iter`, `cuda_peak_MB`.
   Затем замер памяти bs 4: `--iters 20 --bs 4 --accum 1 --warmup-iters 5 --pred-split none`. OOM → `--bs 2 --accum 8`.
   Бюджет полосы GPU0 ≈ s/ит(bs 4) × 4 × ITERS × 3 + сетка; не влезает в 12 ч → `ITERS=6000` или фолды по сессиям.
2. `bash kaggle/rfdetr_folds.sh det runs/rfdetr rfdetr-m 8000 $DINO`. `rfdetr-l` отдельно не гнать: в 1.11.2 при том же `--size` это та же сеть.
3. Скачать `rfdetr_preds.tgz` → машина с данными: по фолдам `coco_preds_to_stack.py --coco det/grid_f$k/valid/_annotations.coco.json --preds runs/rfdetr/f$k/grid/preds_valid.json --group sp_rfdetr --radius 4 --merge-m 4`
   → слияние → `scoreboard.py <флаги A> --add work/sp_rfdetr_oof.geojson:sp_rfdetr --honest 10` и `--replace sp_focal=...`.
4. Приёмка П1–П6 (`/workspace/koz3_shared/EXPERIMENTS.md:13-20`). Пороги — только honest-10/LOFO, не на той же семье.
   Ожидание из DETECTORS_SOTA.md: без предобученного декодера −1…−3 п. F1 к версии с COCO-чекпойнтом. Это не замер.

## Проверено на боксе (CPU, 2 потока, nice 10; не оценка для T4)
| проверка | итог | пик RSS | файл |
|---|---|---|---|
| smoke RF-DETR-M, MAL, 256 px, bs 1, 10 ит. | 2,035 с/ит, всего 25,3 с, перенос DINOv2: 223/223 тензора | 1798 МБ | bot-5/tmp/rf_smoke_m/timing.json, tmp/rf_smoke_m.log |
| то же, `--loss iabce` (родной лосс rfdetr) | 2,427 с/ит | 1751 МБ | tmp/rf_smoke_m_iabce/timing.json |
| smoke RF-DETR-L, 256 px | 1,945 с/ит; 32,98 M пар., лосс тот же, что у M | 1796 МБ | tmp/rf_smoke_l/timing.json |
| smoke M, bs 2 | снят сторожем (RSS 2070208 КБ > 2000 МБ) | >2000 МБ | (вывод run_smoke_rf.sh, 21:57 МСК) |
| `--iters 2 --accum 2` + сохранение, затем `--load` → 103 окна, 10300 рамок, предсказания совпадают | ок | 1901 / 681 МБ | tmp/rf_save, tmp/rf_load |
| формат: smoke `--min-score 0 --tta-flip` → 400 рамок / 4 окна → coco_preds_to_stack | 337 объектов sp_rfdetr | 1769 МБ | tmp/rf_smoke_m0/stack.json |
| перенос ядра патча 14→16: отклик conv на гладких картинках | PI-resize cos 0,99998, отн. ошибка 0,0073 (билинейный: 0,9973 / 0,074) | — | команда в NOLIDAR §11 |
Скорость на боксе плавает с нагрузкой (load avg 10–14): тот же smoke M раньше шёл 9,98 с/ит, с `--tta-flip` — 7,95 с/ит. Ограничение памяти: `ulimit -v 4000000` + сторож RSS 2000 МБ (`work/run_smoke_rf.sh`).
