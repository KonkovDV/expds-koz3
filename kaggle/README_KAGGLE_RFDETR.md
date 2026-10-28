# RF-DETR-M без COCO-чекпойнта на Kaggle — рецепт (bot-5, 28.10.2026)

**Только честный замер. Ничего не идёт в сдачу без отдельного решения** (запасной путь на случай, если письма о весах не будет).
Запуск на Kaggle — bot 9.

Лицензии: пакет `rfdetr==1.11.2` — Apache-2.0 (METADATA, `rfdetr/config.py:737`); энкодер `facebook/dinov2-small` — Apache-2.0
(карточка HF `license: apache-2.0`; github.com/facebookresearch/dinov2/LICENSE — Apache 2.0). Детекторные чекпойнты
`rf-detr-*.pth` (COCO/O365, `rfdetr/assets/model_weights.py:191+`) **не скачиваются**: `pretrain_weights=None`.
Декодер, проектор и головы — с нуля.

Файлы (ветка `bot5-rfdetr-kaggle`):
* `scripts/rfdetr_train.py` — свой цикл обучения; из rfdetr берутся только `build_model_from_config`/`build_criterion_from_config`.
  Что своё: ручной перенос DINOv2-S в энкодер (ядро патча 14→16 псевдообратным ресайзом, позиции 37²→сетка модели бикубикой),
  MAL (`--loss mal`, γ 1,5; подмена `SetCriterion.loss_labels`), flat-cosine (warmup 500 → плато до 50 % → косинус до 5 %),
  последние 15 % без copy-paste/сильных ауг., copy-paste курганов (povr/tselye) вместо Mosaic, EMA 0,999, накопление градиента,
  fp16 (лосс в fp32), послойный lr ViT (×0,8/слой). DN нет в RF-DETR вообще (group-DETR, 13 групп) — DN=0 «бесплатно».
* `kaggle/rfdetr_folds.sh <DET> <OUT> [MODEL] [ITERS] [DINO_DIR]` — 2 полосы (GPU0 f0 f2 f4, GPU1 f1 f3): обучение + сетка фолда.
* `kaggle/rfdetr_kaggle.ipynb` — pip → dinov2-small → smoke → память bs 4 → полосы → `rfdetr_preds.tgz`.

## Шаг 0 — данные
Те же `det/f0..f4`, `det/grid_f0..4`, что для D-FINE (`kaggle/README_KAGGLE.md`, шаг 0; `coco_from_cache.py --layout rfdetr --folds 5 --fold k`).

## Шаг 1 — smoke на T4 (первый шаг; только честный замер)
```bash
pip install "rfdetr==1.11.2" --no-deps && pip install supervision pyDeprecate "pydantic>=2"
python scripts/rfdetr_train.py --data det/f0 --out runs/rf_smoke --model rfdetr-m --size 640 --fp16 --smoke
cat runs/rf_smoke/timing.json   # s_per_iter, cuda_peak_MB; в логе: dinov2: {'copied': 223, ..., 'missing': 0}
python scripts/rfdetr_train.py --data det/f0 --out runs/rf_mem --model rfdetr-m --size 640 --fp16 \
  --iters 20 --bs 4 --accum 1 --warmup-iters 5 --log-every 5 --pred-split none   # память и с/ит при bs 4
```
Бюджет полосы GPU0 ≈ `s_per_iter(bs4) × 4 (accum) × ITERS × 3 + сетка`. Не влезает в 12 ч → `ITERS=6000` или фолды по сессиям; OOM → `--bs 2 --accum 8`.

CPU коробки (2 потока, nice 10, 256 px, bs 1, сторож 2000 МБ): 2,035 с/ит, пик RSS 1798 МБ (`bot-5/tmp/rf_smoke_m/timing.json`).
bs 2 на 256 px превысил 2000 МБ и снят сторожем — на боксе только bs 1. Для T4 это не оценка.

## Шаг 2–3 — обучение и сетка
```bash
bash kaggle/rfdetr_folds.sh det runs/rfdetr rfdetr-m 8000 [DINO_DIR]
```
По умолчанию: 640 px (родное окно 0,5 м; без пересэмплинга), bs 4 × accum 4, AdamW lr 1e-4, энкодер 1,5e-4 × 0,8^(глубина),
wd 1e-4, clip 0,1, EMA 0,999, MAL γ 1,5, copy-paste p 0,5 (1–4 вырезки, поле 30 %), сэмплинг 60 % окон с объектами, вес класса 1/√n (кап 4).
Предсказание: flip-TTA, NMS 0,5, ≤100 рамок/окно, min_score 0,05. `rfdetr-l` в 1.11.2 при одинаковом `--size` — та же сеть, что M
(одинаковые 32,98 M пар. и лосс в smoke), отличается только родным разрешением 704 → отдельно не гонять.
Выход: `runs/rfdetr/f$k/grid/preds_valid.json` (COCO results, px PNG, category_id 1..5) + `timing.json`.

## Шаг 4 — стек и табло (машина с данными)
Как `README_KAGGLE.md` шаг 4, с `--preds runs/rfdetr/f$k/grid/preds_valid.json --group sp_rfdetr`, затем
`scoreboard.py <флаги A> --add work/sp_rfdetr_oof.geojson:sp_rfdetr --honest 10` и `--replace sp_focal=...`.
Приёмка П1–П6 (`/workspace/koz3_shared/EXPERIMENTS.md:13-20`). Пороги — только honest-10/LOFO.
Проверка формата на коробке: smoke `--min-score 0 --tta-flip` → 400 рамок / 4 окна → `coco_preds_to_stack.py --group sp_rfdetr` → 337 объектов (`bot-5/tmp/rf_smoke_m0/stack.json`).
