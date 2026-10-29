# RF-DETR-M (O365→COCO-чекпойнт) на Kaggle — рецепт (bot-5, 29.10.2026)

Запуск на Kaggle — bot 9. Честный замер OOF по 5 семейным фолдам; в сдачу — только после приёмки П1–П6.

**Веса по умолчанию — COCO** (`--init coco`): `rf-detr-medium.pth` = `https://storage.googleapis.com/rfdetr/medium_coco/checkpoint_best_regular.pth`,
md5 `7223f764a87b863f02eb8d52bf0ce2ee` (`rfdetr/assets/model_weights.py:225-229`; совпал у скачанного `bot-5/tmp/rfhome/rf-detr-medium.pth`, 404 992 918 байт).
Грузится штатной `rfdetr.models.weights.load_pretrain_weights` (md5, PE → сетка `--size`, срез групп запросов); голова классов
переинициализируется под K+1 = 6. Лицензии: пакет `rfdetr==1.11.2` и веса N/S/M/L — Apache-2.0; веса COCO разрешены организаторами (29.10).
Запасные режимы: `--init dinov2` (только энкодер DINOv2-S, как в b355e2c) и `--init scratch`.

### Objects365 (разрешены 29.10) — какой чекпойнт и что проверено
| вариант | чекпойнт | источник | md5 | лицензия | статус |
|---|---|---|---|---|---|
| **A (основной)** RF-DETR-M, O365→COCO | `rf-detr-medium.pth` | URL выше | `7223f764a87b863f02eb8d52bf0ce2ee` | Apache-2.0 | скачан, md5 ок, smoke ок |
| B (необяз.) RF-DETR-Base, чистый O365 | `rf-detr-base-o365.pth` | `https://storage.googleapis.com/rfdetr/top-secret-1234/lwdetr_dinov2_small_o365_checkpoint.pth` | `d93f4921ccbb0f0a2e4364bed290892b` | Apache-2.0 по реестру пакета `rfdetr` (не Plus/PML: PML 1.0 только у A/F/P/XL/2XL, README rf-detr:21,61-71); отдельной строки Base в таблице лицензий README нет | скачан (384 452 411 байт), md5 ок, smoke ок |
| запасной | `--init dinov2` / `--init scratch` | — | — | — | smoke ок |

* **Чистого O365-чекпойнта размера M нет**: в реестре `rfdetr/assets/model_weights.py` (1.11.2) O365-only есть только `rf-detr-base-o365.pth` (стр. ~195-199).
* Что M уже предобучен на O365 — **не проверено нами**, это утверждение авторов: arXiv 2511.09554, табл. 5 («+ Additional O365 Pre-Training»), прил. A (O365 с псевдометками SAM2), и `research/PRETRAINED_WEIGHTS_2026-10-29.md:36` («Objects365 → COCO»). Поэтому основной — A.
* md5 обоих файлов совпал с реестром и с `x-goog-hash` GCS (`curl -sI`, 29.10).
* B: Base = DINOv2-S windowed, патч 14, сторона кратна 56 → `MODEL=rfdetr-b SIZE=672`; голова классов 365 → 6 (лог rfdetr). Запускать после A, отдельной сессией; оценка T4 — как у A ×(672/640)² ≈ ×1,1 (оценка, не замер).


Файлы (ветка `bot5-rfdetr-coco`, от `bot5-rfdetr-kaggle` b355e2c):
* `scripts/rfdetr_train.py` — свой цикл: MAL γ 1,5, flat-cosine (warmup 500 → плато 50 % → косинус до 5 %), последние 15 % без copy-paste,
  copy-paste курганов, EMA 0,999, accum, fp16 (лосс fp32), послойный lr ViT ×0,8. Новые флаги: `--init coco|dinov2|scratch`, `--coco-weights PATH`.
* `kaggle/rfdetr_folds.sh <DET> <OUT> [MODEL] [ITERS] [DINO_DIR]`, env `INIT` (coco), `COCO_W` — 2 полосы (GPU0 f0 f2 f4; GPU1 f1 f3).
* `kaggle/requirements_rfdetr.txt` — transformers==5.21.0, timm 1.0.30, pydantic 2.14.1, supervision 0.31.3, pyDeprecate 0.9.0.
* `kaggle/rfdetr_kaggle.ipynb` — pip → чекпойнт + md5 → smoke → память bs 4 → полосы → `rfdetr_preds.tgz`.

## Входы (Kaggle Datasets)
* `expds-koz3-bot5` — этот репо, ветка `bot5-rfdetr-coco`.
* `koz3-spor-det05` — `f0..f4`, `grid_f0..grid_f4` из `coco_from_cache.py --layout rfdetr --folds 5 --fold k` (те же, что для D-FINE, `kaggle/README_KAGGLE.md` шаг 0).
* Интернет ON (pip + чекпойнт), либо Dataset с `rf-detr-medium.pth` и `COCO_W=<путь>`.

## Шаг 1 — smoke на T4 (первый)
```bash
pip install --no-deps "rfdetr==1.11.2" && pip install -r kaggle/requirements_rfdetr.txt
python scripts/rfdetr_train.py --data det/f0 --out runs/rf_smoke --model rfdetr-m --size 640 --fp16 --init coco --coco-weights $COCO_W --smoke
# в логе: coco: {... 'md5': '7223f764a87b863f02eb8d52bf0ce2ee', 'tensors': 510, 'changed': 508, 'head_out': 6}
python scripts/rfdetr_train.py --data det/f0 --out runs/rf_mem --model rfdetr-m --size 640 --fp16 --init coco --coco-weights $COCO_W \
  --iters 20 --bs 4 --accum 1 --warmup-iters 5 --log-every 5 --pred-split none    # s/ит и cuda_peak_MB при bs 4
```
OOM → `--bs 2 --accum 8`. Бюджет полосы GPU0 = `s_per_iter(bs4) × 4 × ITERS × 3 + сетка`; > ~11 ч → `ITERS=6000` или фолды по сессиям.

## Шаг 2 — обучение + сетка
```bash
INIT=coco COCO_W=$COCO_W bash kaggle/rfdetr_folds.sh det runs/rfdetr rfdetr-m 8000
```
640 px, bs 4 × accum 4, AdamW lr 1e-4 / энкодер 1,5e-4, flip-TTA, NMS 0,5, ≤100 рамок/окно, min_score 0,05.
`rfdetr-l` с COCO — другой чекпойнт (`rf-detr-large-2026.pth`, md5 `5cb72153541cbcb9aa6efa26222acc75`) и другая голова декодера; не в первой очереди.

## Выходы
`runs/rfdetr/f$k/grid/preds_valid.json` (COCO results, px PNG, category_id 1..5), `f$k/timing.json`, `f$k/model/{model.pt,rf_config.json}`,
логи `f$k.train.log`, `f$k.grid.log`; всё в `rfdetr_preds.tgz` (последняя ячейка).

## Шаг 3 — стек и табло (машина с данными)
По фолдам `coco_preds_to_stack.py --coco det/grid_f$k/valid/_annotations.coco.json --preds runs/rfdetr/f$k/grid/preds_valid.json --group sp_rfdetr --radius 4 --merge-m 4`
→ слияние → `scoreboard.py <флаги A> --add work/sp_rfdetr_oof.geojson:sp_rfdetr --honest 10` и `--replace sp_focal=...`. Приёмка П1–П6
(`/workspace/koz3_shared/EXPERIMENTS.md:13-20`), пороги только honest-10/LOFO.

## T4-часы
Замера на T4 нет. План: 9 GPU-ч на сессию S2 RF-DETR-M (`/workspace/koz3_shared/DETECTORS_SOTA.md:302`, оценка плана). Точное — по формуле выше после smoke.

## Проверено на боксе (CPU, 2 потока, nice 10, сторож 2000 МБ, синтетика `tmp/smoke_coco` 256 px; это не оценка T4)
| проверка | итог | пик RSS | файл |
|---|---|---|---|
| `--init coco --smoke --bs 1 --accum 1` | 1,77 с/ит; лосс 23,70 → 12,71 за 10 ит.; заменено 508/510 тензоров; md5 ок | 1784 МБ | bot-5/tmp/rf_coco_bs1/timing.json |
| контроль `--init dinov2`, то же | 1,40 с/ит; лосс 17,73 → 17,18 | 1805 МБ | bot-5/tmp/rf_dinov2_bs1/timing.json |
| вариант B `--model rfdetr-b --size 280 --init coco --coco-weights rf-detr-base-o365.pth --smoke --bs 1 --accum 1` | 1,5–1,8 с/ит; лосс 19,97 → 10,71; заменено 486/488 тензоров; head_out 6 | 1627 МБ (сторож) | bot-5/tmp/rf_o365b_bs1.log |
| `--iters 2 --accum 2` + сохранение → `--load` → 400 рамок / 4 окна | ок | 1844 / 678 МБ (сторож) | bot-5/tmp/rf_coco_save.log, tmp/rf_coco_load.log |
Smoke с bs по умолчанию на этой синтетике снят сторожем (>2000 МБ) и при `--init dinov2` тоже (`tmp/rf_dino_ctrl.log`) — дело в батче, не в весах.
