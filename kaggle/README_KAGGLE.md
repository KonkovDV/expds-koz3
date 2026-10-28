# D-FINE-L / RT-DETRv2 на Kaggle — рецепт (bot-5, 28.10.2026)

**Только честный замер. Ничего не идёт в сдачу до письма организаторов** (лицензия весов `ustc-community/dfine-large-coco`,
`PekingU/rtdetr_v2_r50vd` — проверить в карточке HF; AI_WORK_PLAN 2.15). Запуск на Kaggle — bot 9.

Файлы (ветка `bot5-dfine-kaggle`):
* `scripts/det_train_hf.py` — свой тренер на `transformers.AutoModelForObjectDetection` (код конкурентов не использовался).
* `kaggle/dfine_folds.sh` — 5 фолдов, 2 полосы GPU (GPU0: f0 f2 f4, GPU1: f1 f3), обучение + предсказание сетки фолда.
* `kaggle/dfine_kaggle.ipynb` — ноутбук: pip → smoke → фолды → архив предсказаний.

## Шаг 0 — данные (машина с полными данными, CPU)
```bash
python -m expds.tiles --root <DATA>/train-expds-3 --gt work/gt_train_v2.geojson --markup sp_or \
  --out cache/spor_det05 --res 0.5 --tile 640 \
  --classes kurgany_povrezhdennye kurgany_tselye gorodishcha arkhitektury fortifikatsii
python -m expds.tiles --root <DATA>/train-expds-3 --gt work/gt_train_v2.geojson --markup sp_or \
  --out cache/spor_det05_grid --res 0.5 --tile 640 --grid-overlap 0.25 \
  --classes kurgany_povrezhdennye kurgany_tselye gorodishcha arkhitektury fortifikatsii
for k in 0 1 2 3 4; do
  python scripts/coco_from_cache.py --cache cache/spor_det05      --out det/f$k      --layout rfdetr --images --folds 5 --fold $k
  python scripts/coco_from_cache.py --cache cache/spor_det05_grid --out det/grid_f$k --layout rfdetr --images --folds 5 --fold $k
done
```
`det/` → Kaggle Dataset `koz3-spor-det05`; репо → Dataset `expds-koz3-bot5`. (В `grid_f$k` нужна только `valid/`: это семьи фолда k, т. е. OOF.)
Кэш, собранный до 27.10, CRS окна не хранит — собирать заново.

## Шаг 1 — smoke на T4 (первый шаг bot 9)
```bash
python scripts/det_train_hf.py --data det/f0 --out runs/smoke --model dfine-l --size 640 --fp16 --smoke
cat runs/smoke/timing.json      # s_per_iter, cuda_peak_MB, pred_s
```
Бюджет полосы GPU0 ≈ `s_per_iter × ITERS × 3 + сетка`. Не влезает в 12 ч — `ITERS=6000` или фолды по сессиям.
На CPU коробки (2 потока, 256 px, bs 2) smoke прошёл: 12,19 с/ит, пик RSS 1577 МБ, 400 рамок на 4 окна (`bot-5/tmp/dfl_smoke/timing.json`). Для T4 это не оценка.

## Шаг 2–3 — обучение и сетка
```bash
bash kaggle/dfine_folds.sh det runs/dfine dfine-l 8000
# RT-DETRv2 — тем же скриптом: bash kaggle/dfine_folds.sh det runs/rtv2 rtdetrv2-r50 8000
```
Гиперпараметры по умолчанию: AdamW lr 1e-4, backbone ×0,1, wd 1e-4, OneCycle pct 0,05, clip 0,1, fp16, bs 8, 640 px;
сэмплинг 60 % окон с объектами, вес класса 1/√n (кап 4); ауг. flip/rot90/яркость/гамма/серый/blur (`--strong-aug` — порча качества).
Предсказание: flip-TTA, NMS IoU 0,5, ≤100 рамок/окно, min_score 0,05. Если loss NaN/коллапс — `--num-dn 0`.
Выход: `runs/dfine/f$k/grid/preds_valid.json` (COCO results, px PNG, category_id 1..5) + `timing.json`.

## Шаг 4 — стек и табло (машина с данными)
```bash
for k in 0 1 2 3 4; do
  python scripts/coco_preds_to_stack.py --coco det/grid_f$k/valid/_annotations.coco.json \
    --preds runs/dfine/f$k/grid/preds_valid.json --group sp_dfine --radius 4 --merge-m 4 --out work/sp_dfine_f$k.geojson
done
python -c "import json;F=[f for k in range(5) for f in json.load(open(f'work/sp_dfine_f{k}.geojson'))['features']];json.dump({'type':'FeatureCollection','features':F},open('work/sp_dfine_oof.geojson','w'),ensure_ascii=False)"
python scripts/scoreboard.py <флаги базы A> --add work/sp_dfine_oof.geojson:sp_dfine --honest 10 --out dfine_add.json
python scripts/scoreboard.py <флаги базы A> --replace sp_focal=work/sp_dfine_oof.geojson --honest 10 --out dfine_rep.json
```
Приёмка — П1–П6 из `/workspace/koz3_shared/EXPERIMENTS.md:13-20` (honest-10 Δ ≥ +0,002; бутстреп семей P ≥ 0,85;
LOFO > 0 и ≥ 4/5 случайных раскладок; povr/tselye/gor ≥ −0,0005, new34/optical ≥ −0,001; Δ < 0,006 — только пакетом).
Проверка формата на коробке: smoke-предсказания → `coco_preds_to_stack.py` → 68 объектов группы sp_dfine (`bot-5/tmp/dfl_smoke/stack_sp_dfine.geojson`).
