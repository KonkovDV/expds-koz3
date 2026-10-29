# Для bot 9: RF-DETR-M (O365→COCO-чекпойнт) на Kaggle 2×T4 (bot-5, 29.10.2026)

Готово к очереди GPU. Веса по умолчанию: COCO `rf-detr-medium.pth` (Apache-2.0, md5 `7223f764a87b863f02eb8d52bf0ce2ee`, `rfdetr/assets/model_weights.py:225-229`).
**Objects365 (разрешён 29.10):** чистого O365-чекпойнта размера M нет (реестр rfdetr 1.11.2). По статье авторов (arXiv 2511.09554, табл. 5, прил. A)
M уже предобучен O365 → COCO — это их утверждение, нами не проверено; поэтому основной вариант A = `rf-detr-medium.pth`.
Вариант B (необязательно, после A): ближайший размер с чистым O365 — RF-DETR-Base `rf-detr-base-o365.pth`,
`https://storage.googleapis.com/rfdetr/top-secret-1234/lwdetr_dinov2_small_o365_checkpoint.pth`, md5 `d93f4921ccbb0f0a2e4364bed290892b`
(совпал с реестром и `x-goog-hash` GCS; скачан на бокс, smoke ок: 486/488 тензоров, RSS 1627 МБ, `tmp/rf_o365b_bs1.log`), Apache-2.0 по пакету rfdetr (не Plus/PML).
Запуск B: `INIT=coco COCO_W=…/rf-detr-base-o365.pth SIZE=672 bash kaggle/rfdetr_folds.sh DET OUT_B rfdetr-b 8000` (ячейка в конце ноутбука). Коммит c43cb08.
Запасной: `--init dinov2`.
Прежняя версия «без COCO» лежит в `tmp/FOR_BOT9_RFDETR.nococo.bak.md`, режим `--init dinov2`.

* Код: ветка `bot5-rfdetr-coco` (от `bot5-rfdetr-kaggle` b355e2c, не запушена), рабочая копия `/workspace/koz3_shared/bot-5/wt_rf`.
* Ноутбук: `kaggle/rfdetr_kaggle.ipynb`. Рецепт: `kaggle/README_KAGGLE_RFDETR.md`, в нём шаги, входы, выходы и проверки на боксе.
* Входы: Dataset `expds-koz3-bot5` (репо на этой ветке), Dataset `koz3-spor-det05` (`f0..f4`, `grid_f0..4`, те же, что для D-FINE), Internet ON.
  Без интернета нужен Dataset с `rf-detr-medium.pth` и `COCO_W`.
* Порядок:
  1. Smoke, затем замер bs 4. В логе должно быть `coco: {... 'changed': 508, 'head_out': 6}`.
  2. Обучение: `INIT=coco COCO_W=… bash kaggle/rfdetr_folds.sh DET OUT rfdetr-m 8000`.
  3. Скачать `rfdetr_preds.tgz`.
  4. `coco_preds_to_stack --group sp_rfdetr`, затем `scoreboard --honest 10` и П1–П6.
* Выходы: `rfdetr/f$k/grid/preds_valid.json` (COCO results, px, cat 1..5), `timing.json`, `model/`, логи. Всё упаковано в `rfdetr_preds.tgz`.
* T4-часы: замера нет. План — 9 GPU-ч на сессию (`/workspace/koz3_shared/DETECTORS_SOTA.md:302`), это оценка.
  Точная цифра: `s_per_iter(bs4) × 4 × 8000 × 3 + сетка` на полосу GPU0 после smoke. Если выходит больше ~11 ч, ставить `ITERS=6000`.
* Бокс (CPU, 2 потока, bs 1, 256 px, синтетика): 1,77 с/ит, лосс 23,70 → 12,71 за 10 ит., RSS 1784 МБ (`tmp/rf_coco_bs1/timing.json`).
  Сохранение и `--load` проверены (`tmp/rf_coco_load.log`).
