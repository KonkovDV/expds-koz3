# MataNerdy — вердикт (v2, приоритетная проверка KOZ2 vs KOZ3)

**НЕ ДОВЕРЯТЬ 0.66–0.75 как KOZ3-числу.** Это работа эпохи КОЗ №2: KOZ2-метрика (своя порт-версия), KOZ2-датасет, 5 классов из 7, плюс пулинг пиксельных координат патчей.
Честная KOZ3-оценка их модели неизвестна без прогона; потолок 0.80 (нет 2 классов), реалистично ≤ нашего 0.315.

## Какой конкурс / датасет
- Один коммит 52b3854, `Thu Jun 25 13:50:38 2026 +0400` (`git log`). Данные KOZ3 в S3 `train-expds-3` — last_modified 2026-09-30 (/workspace/data/s3_listing.jsonl), т.е. позже коммита.
- В репо нет упоминаний KOZ/expds (grep), официального compute_metrics.py не поставляют.
- Датасет локальный: `01_geodata_to_cv/build_multiclass_dataset.py:25` DATASET_ROOT=/Volumes/Lexar/Датасет/; Kaggle: `matanerdy/arkheologic` (notebook all-class-baseline-competition-metrics.ipynb cell 1), `matanerdy/kurgans-dataset` (run_kaggle_experiments.sh:4).
- Регионы — старая нумерация/верхний регистр: из 104 регионов их split (train/val_split.csv) только 82 совпадают по имени с нашими 210 в train-expds-3 (напр. 004_ДЕМИДОВКА↔015_Демидовка, 005_ЛУБНО↔003_Лубно); 22 не совпадают (017_ВЫШЕГЖА, 019_ОСЕЧКИ_1, 043_ТОРУ_АЙГЫР…). ⇒ KOZ2-набор (train-expds-2), не KOZ3.

## Какая метрика (diff их vs koz3-1.5.0)
| | их `03_.../utils/polygon_metrics.py` | KOZ2 официальная (recara `compute_metrics.py`) | KOZ3 1.5.0 `/workspace/expds-koz3/materials/compute_metrics.py` |
|---|---|---|---|
| веса | :22-28 27.8/22.2/16.7/11.1/5.6 (5 кл.) | :50-58 те же + object_poly 11.1, finds_points 5.5 | :83-91 30/20/15/10/10/5/10 |
| матч | :155 IoU>0.3 **ИЛИ** центроид в GT | :406 IoU>0.1 ИЛИ центроид | :923-928 cov=∩/min(area)≥0.5 **И** inner-point в GT |
| FP | :92 tp/(tp+fp) | :285 tp/(tp+fp) | :696 tp/(tp+0.5·fp) |
| координаты | :50-72 пиксели патча, :115-128 region_name игнорируется ⇒ все патчи в одной системе | — | реальные UTM по региону |
Веса 1:1 = KOZ2 (без object_poly/finds_points) ⇒ метрика — порт KOZ2 с порогом IoU 0.3 (как у KateBuikina KOZ2 metric.py:24-30, :104; см. bot-7/NOTES_KateBuikina-expds-solution.md:7).

## Классы
5: kurgany_povrezhdennye, kurgany_tselye, gorodishcha, arkhitektury, fortifikatsii (overlay_5_classes.py:25-32). Нет object_poly и finds_points ⇒ на полном KOZ3 GT максимум 80/100 веса (compute_metrics.py:83-91).

## Оценка под правилами KOZ3 (дешёвый CPU-пересчёт)
`mata_sanity/koz3_rules_sim.py` → `koz3_rules_sim.out`: тот же «слепой» предиктор (диск r=4px в центре патча, класс по приору val; без изображения), 601 val-патч Li/Ae/SpOr, GT = целевой эллипс:
| протокол | их метрика | официальная KOZ3 |
|---|---|---|
| пулинг координат (как у них) | 0.9374 | 0.9572 |
| патчи разнесены (честно) | 0.2263 | 0.2824 |
| все = povrezhdennye | 0.2170 | 0.2957 |
Вывод: сами правила KOZ3 (FP×0.5) на их протоколе дают **не ниже**, разница метрик — не главное. Главное — протокол: объектно-центрированные кропы + пулинг пиксельных координат (слепой предиктор → 0.94), отбор чекпойнта/seed/72 конфигов постпроцесса по val, нет инференса по полному растру (фон → FP), нет 2 классов (−20% веса), нет вывода в EPSG:3857.
Числовой честной оценки KOZ3 нет (нужен прогон на full-raster с family-фолдами); по аргументам выше — ожидаемо ≤0.315 (наш honest), потолок 0.80.

Предыдущая версия: mata_sanity/MATANERDY_VERDICT.v1.bak. Сим пулинга: mata_sanity/pooled_metric_sim.out.
