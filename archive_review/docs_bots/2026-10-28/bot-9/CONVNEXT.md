# CONVNEXT.md — ядра Kaggle: ConvNeXt-T U-Net по лидару (KOZ3 №3)

Статус: **СОБРАНО, НЕ ЗАПУШЕНО** (28.10.2026, ~22:45 МСК). Папка: `/workspace/koz3_shared/bot-9/kaggle/convnext_lidar/`.
Основа: DETECTORS_SOTA.md, bot-10/REPORT5_lidar59.md, COMMON.md; голова центров — bot-2 `centerhead` @ 788202a.
dfine_spor и nolidar_verifier не трогал.

## 1. Устройство
- `src/convnext_main.py` (ядро), `src/cnx_head.py` (адаптер голов), `src/merge_oof.py`, `src/validate_oof.py`,
  `build.py` встраивает в каждое ядро код, CNX_MODE/HEAD/JOBS/REGIONS_DEFAULT, FOLDS и вендор (base64 + манифест sha256)
  по маркеру `# @@EMBED@@`. Один .py на ядро, 876 КБ (`du -k koz3-cnx-li-*/*.py`).
- Рецепт: кроп 512, bs 8, 12 000 ит., lr 3e-4, wd 1e-4, warmup 500, EMA 0.999, AMP fp16, copy-paste, D8, obj_frac 0.6
  (CFG в logs/dry_prep.log); инференс тайлами `tile_origins` + D8-TTA, пиковый decode.
- Голова по умолчанию `centerhead` (bot-2): stride 4 (`decoder.blocks[2]`, 64 канала, 128×128 при 512), hm 2 класса,
  wh (1 канал log-радиус), off; лосс `center_loss(...)["loss_center"] + mask_loss`; маски городищ/укреплений/arkhitektury —
  seg-голова полного разрешения; круг-выход 0.5 м. Подробно — `convnext_lidar/HEAD_INTERFACE.md` и §8 ниже.
- Фолды: честные семейные (5 фолдов, LOFO по семьям), 59 участков, {0: 6, 1: 1, 2: 13, 3: 18, 4: 21} (logs/check_maps.log).
- Варианты 2×2: c3 = lrm_s, lrm_l, slope; c5 = c3 + svf_r, opp_r (rvt-py `--no-deps`) × ImageNet (`-in`) / scratch (`-sc`).
  **ImageNet — только для измерения, не для сабмита, до письма организаторов о весах.**
- Метрика: официальный compute_metrics.py (koz3-1.5.0), матчинг один-к-одному. В логе ядра печатается `score(Li-only)` —
  это НЕ число scoreboard; честное — у бота 10 (`scoreboard --honest 10`, П1–П5) по OOF-geojson.
- Защиты: `--require-gpu` по умолчанию (`--allow-cpu` только на боксе), закреплённый pip + жёсткая проверка импортов,
  проверка sha вендора, ранний выход при ошибке. S3: бакет `train-expds-3`, `https://s3.ru-7.storage.selcloud.ru`.

## 2. Ядра (slug = dmitrijkv/<папка>)
| папка | задания | GPU | зависимости |
|---|---|---|---|
| koz3-cnx-li-prep | S3 LAS → каналы рельефа `cnx_relief/*.npy` (все 59) | нет (CPU) | dataset koz3-s3-cred |
| koz3-cnx-li-smoke | c5-in:4, c3-sc:3; 10 ит., 2 участка | T4 | dataset; рельеф строит сам |
| koz3-cnx-li-c3sc-f01 / -c3sc-f23 | c3-sc фолды 0,1 / 2,3 | 2×T4 | dataset + kernel koz3-cnx-li-prep |
| koz3-cnx-li-c5sc-f01 / -c5sc-f23 | c5-sc фолды 0,1 / 2,3 | 2×T4 | то же |
| koz3-cnx-li-c3in-f01 / -c3in-f23 | c3-in фолды 0,1 / 2,3 | 2×T4 | то же |
| koz3-cnx-li-c5in-f01 / -c5in-f23 | c5-in фолды 0,1 / 2,3 | 2×T4 | то же |
| koz3-cnx-li-c3-f4 / -c5-f4 | c3-in:4 + c3-sc:4 / c5-in:4 + c5-sc:4 | 2×T4 | то же |
Метаданные: `kernel-metadata.json` в каждой папке (machine_shape NvidiaTeslaT4, dataset_sources ["dmitrijkv/koz3-s3-cred"],
kernel_sources ["dmitrijkv/koz3-cnx-li-prep"] у полных ядер). По заданию на GPU (2 полосы, по одному на T4).

## 3. Команды пуша (по одной, следующее — после COMPLETE предыдущего; одно GPU-ядро за раз)
```
cd /workspace/koz3_shared/bot-9/kaggle/convnext_lidar
KAGGLE_API_TOKEN=$(cat ~/.kaggle/access_token) ~/.local/bin/kaggle kernels push -p koz3-cnx-li-prep      # CPU, можно параллельно со smoke
KAGGLE_API_TOKEN=$(cat ~/.kaggle/access_token) ~/.local/bin/kaggle kernels push -p koz3-cnx-li-smoke
KAGGLE_API_TOKEN=$(cat ~/.kaggle/access_token) ~/.local/bin/kaggle kernels push -p koz3-cnx-li-c3sc-f01
KAGGLE_API_TOKEN=$(cat ~/.kaggle/access_token) ~/.local/bin/kaggle kernels push -p koz3-cnx-li-c3sc-f23
KAGGLE_API_TOKEN=$(cat ~/.kaggle/access_token) ~/.local/bin/kaggle kernels push -p koz3-cnx-li-c3-f4
KAGGLE_API_TOKEN=$(cat ~/.kaggle/access_token) ~/.local/bin/kaggle kernels push -p koz3-cnx-li-c5sc-f01
KAGGLE_API_TOKEN=$(cat ~/.kaggle/access_token) ~/.local/bin/kaggle kernels push -p koz3-cnx-li-c5sc-f23
KAGGLE_API_TOKEN=$(cat ~/.kaggle/access_token) ~/.local/bin/kaggle kernels push -p koz3-cnx-li-c5-f4
KAGGLE_API_TOKEN=$(cat ~/.kaggle/access_token) ~/.local/bin/kaggle kernels push -p koz3-cnx-li-c3in-f01   # ImageNet — только замер
KAGGLE_API_TOKEN=$(cat ~/.kaggle/access_token) ~/.local/bin/kaggle kernels push -p koz3-cnx-li-c3in-f23
KAGGLE_API_TOKEN=$(cat ~/.kaggle/access_token) ~/.local/bin/kaggle kernels push -p koz3-cnx-li-c5in-f01
KAGGLE_API_TOKEN=$(cat ~/.kaggle/access_token) ~/.local/bin/kaggle kernels push -p koz3-cnx-li-c5in-f23
```
Полные ядра нужно пушить после COMPLETE prep (они читают его выход через kernel_sources). `-f4` ядра содержат и `-in`, и `-sc`.
Пересборка после изменений: `python3 build.py` (смена головы bot-2: `CNX_CENTERHEAD_SHA=<sha> python3 build.py`).

## 4. Секреты и данные
Kaggle Secrets: `KOZ3_S3_ACCESS_KEY_ID`, `KOZ3_S3_SECRET_ACCESS_KEY` → иначе env → иначе `/kaggle/input/**/cred.json`
(датасет `dmitrijkv/koz3-s3-cred`). В коде/логах ключей нет: `grep -rlF` значений ключей по bot-9 (без s3cache/venv) →
`files_with_keys=0`, по venv ядра → 0 (команда в STATUS.md, 28.10 ~22:35 МСК).

## 5. pip (чистый venv, `_scratch/venv`, logs/pip_check.log, logs/venv_freeze.txt)
Python 3.13.5; torch 2.11.0+cpu (на Kaggle — cu128 из образа), torchvision 0.26.0, numpy 2.1.3, timm 1.0.29,
segmentation_models_pytorch 0.6.0, rasterio 1.4.3, pyproj 3.7.1, shapely 2.1.1, boto3 1.40.61, scipy 1.16.3, laspy 2.7.0,
rvt-py 2.2.1. `pip check` rc=1 только из-за «rvt-py 2.2.1 requires gdal / matplotlib» — ожидаемо (`--no-deps`, используемые
функции rvt их не импортируют; импорт-проверка ядра проходит: «вендор OK … rvt 2.2.1», logs/dry_prep.log).
Образ Kaggle закреплён: `gcr.io/kaggle-private-byod/python@sha256:9a6bcfda…f9435`.

## 6. Проверки на боксе (CPU, 2 потока, nice -n 10, `--allow-cpu`)
- Сборка: logs/build.log — OK; вендор centerhead = `git archive 788202a`, diff с деревом bot-2 = 0.
- Тест голов: logs/test_head.log rc=0 (d8/untta Δ = 0, decode 8/8, лосс конечен) — HEAD_INTERFACE.md §5.
- Сухой прогон centerhead (S3 → рельеф → 3 ит. → инференс → OOF + карты): logs/dry_run_centerhead.log, exit 0, 32.0 с;
  c5-in_f4 на 016_Хога — 1669 детекций, карты 0.4 МБ, score_li 0.0349; c3-sc_f3 на 191_Осечки_1.3км — 448 детекций,
  карты 0.1 МБ; пик RSS 1 880 344 КБ (logs/dry_run_centerhead_rss.txt).
- Сухой прогон heat: logs/dry_run_heat.log, exit 0, 34.2 с; 942 и 242 детекции; пик RSS 1 559 940 КБ.
- Путь prep → full: prep на 2 участках OK, `016_Хога.npy` побайтно равен smoke-сборке, затем smoke-из-prep c centerhead —
  «ИТОГ: все задания OK», 524 детекции; пик RSS 1 416 152 КБ (logs/dry_prep.log, logs/dry_prep_rss.txt,
  logs/dry_prep_centerhead.log). Ранее prep-манифест хранил полный sha вендора → падал после правки головы; теперь
  сверяется только RELIEF_SHA (старый prep пересобран).
- OOF: logs/validate_oof_v2.log — «ошибки схемы: нет» во всех 4 OOF; контроль идеальных кругов r = 0.5 м
  0.9999994444 (18 GT) / 0.9999991667 (3 GT). Карты: logs/check_maps.log — decode bot-2 по npz = OOF (428/428, 1669/1669).
- Слияние: `src/merge_oof.py` → logs/merge_test.log (40 000 объектов, поля stack_arch; дубликат участка → rc=1).
- s/it на боксе шумный (0.49–6.5 с, CPU) — не оценка для T4.

## 7. Оценка времени (ОЦЕНКА, не замер; из bench_t4/validity.json)
- ConvNeXt-T U-Net fp16 1024²: 59.8–61.2 мс/тайл, D8 499–516 мс; fp32 126.6–128.8 мс; троттлинг sw_power_cap, 70 Вт.
- Обучение 512², bs 8: прямой ≈ 8 × 60/4 ≈ 120 мс; прямой+обратный ≈ ×3 → ≈ 0.37–0.45 с/ит с загрузчиком →
  12 000 ит. ≈ 1.2–1.5 ч на задание. Голова stride 4 дешевле heat (меньше выход), в расчёт не закладывал.
- Инференс D8: ≈ 0.5 с на Мпикс → тест-фолд ≤ 21 участок × ≤ 10 Мпикс (bot-2/RELIEF.md) ≲ 2 мин на задание.
- На ядро: 2 задания параллельно на 2×T4 + S3/кеш/старт → **≈ 2 ч**; 10 полных ядер ≈ **20 ч**, smoke ≈ 0.2–0.5 ч,
  prep — CPU (по квоте GPU не идёт; время зависит от S3 и rvt, оценка 1–3 ч). При учёте квоты по сессии укладывается
  в ~30 ч/нед.; ImageNet-четвёрку можно отложить на следующую неделю. Ограничение 12 ч на ядро — с запасом
  (`CNX_DEADLINE_H`, `CNX_INFER_RESERVE_H` обрезают обучение, чтобы успеть инференс).

## 8. Интерфейс головы (кратко; полностью — convnext_lidar/HEAD_INTERFACE.md)
`build(ch, encoder, weights)` → dict `hm/wh/off/mask`; `targets(objs[x,y,diam_px,cls] после D8, S)` → цели на S/stride;
`loss(out, tgt, valid)` = `center_loss["loss_center"]` (+ `mask_loss` в ядре); `untta(o, k, fl)`;
`decode(maps, left, top, step·stride, valid_h, suppress_h)` → X, Y, cls, score, radius_m. OOF-карты
`maps/maps_cnx_{tag}_{region}.npz` (hm f32, wh/off f16, mask u8, valid, meta с left/top/step/crs) + `region_fold.json`.
Ответ боту 2 по п.5 INTERFACE.md: п.1–4, 6–9 — подтверждены; п.5 — да, но hm в float32 (f16 даёт ≈ 1 % ложных пиков);
п.10 — SpOr в этом ядре нет (стыковка у бота 10).

## 9. Риски
1. rvt (svf/opp) на самом большом LAS (~6 ГБ, bot-2/RELIEF.md) — память prep на Kaggle; на боксе проверено только на 2 малых участках.
2. Пик RSS на боксе 1.88 ГБ при кропе 128 — у предела 2 ГБ; на Kaggle памяти больше, но аккумуляторы участка растут с площадью.
3. Объём OOF-карт: на участок ≤ 10 Мпикс ≈ 30 МБ маски u8 + 5 МБ hm + ≈ 4 МБ wh/off + valid до сжатия (расчёт); выход ядра — сотни МБ.
4. Лимит 20 000 детекций на участок (`CNX_MAX_DET`) — может обрезать хвост на больших участках.
5. DataLoader на 4 vCPU (2 worker на полосу), copy-paste и D8 на CPU — может стать узким местом (с/ит > оценки).
6. Две полосы в одном процессе — GIL; при росте с/ит выносить в процессы.
7. Зависимость от prep через kernel_sources: полные ядра падают рано, если prep не COMPLETE или собран другим кодом каналов.
8. Снимок головы bot-2 заморожен на 788202a; новые коммиты — только пересборкой с `CNX_CENTERHEAD_SHA`.
9. Аугментация высоты для svf/opp — приближение (каналы считаются до аугментации).
10. `score_li` в логе — не число scoreboard; итог — у бота 10.
11. Троттлинг T4 (sw_power_cap, SM 840–1125 МГц в бенче) — время может быть больше оценки.
12. ImageNet-веса — не для сабмита до письма организаторов.

Ничего не пушил, на Kaggle/GitHub не запускал, сообщений не отправлял.

## Готовность к пушу (22:40 МСК, бот 9)
koz3-cnx-li-smoke готов к пушу сразу после COMPLETE koz3-dfine-spor-full (фолды 4,2,1, запущен ботом 1 в 22:40 МСК) — при условии, что koz3-cnx-li-prep (запущен 22:27 МСК) в статусе COMPLETE. Изменений в smoke после CPU-драйрана (logs/dry_prep_centerhead.log) не было.
Команда: `cd /workspace/koz3_shared/bot-9/kaggle/convnext_lidar && KAGGLE_API_TOKEN=$(cat ~/.kaggle/access_token) ~/.local/bin/kaggle kernels push -p koz3-cnx-li-smoke`
На этой неделе после smoke — только «с нуля»: c3sc-f01 → c3sc-f23 → c3-f4 → c5sc-f01 → c5sc-f23 → c5-f4 (решение бота 1). ImageNet (c3in/c5in) — после письма организаторов / на следующей неделе.
