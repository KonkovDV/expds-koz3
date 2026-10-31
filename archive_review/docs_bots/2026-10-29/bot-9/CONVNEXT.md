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

## 28.10.2026 ~23:59 МСК — контракт весов для bot-8 и бенч bot-8 в smoke (пересборка; команды пуша НЕ изменились)
- **Контракт весов.** Каждый фолд дополнительно пишет `/kaggle/working/weights_cnx_{variant}_f{fold}.pt` (weights_only, arch-dict, fp16). `ckpt_cnx_*.pt` остался как был. Описание: `CNX_CONTRACT_FOR_BOT8.md`. Тест: `kaggle/convnext_lidar/tests/test_contract.py` → `kaggle/convnext_lidar/logs/test_contract.json` (max|Δ| = 0).
- **Бенч bot-8 в smoke.** Это «Шаг 11», необязательный последний шаг только в koz3-cnx-li-smoke.
  - build.py упаковывает `git archive` чистой ветки **bot8-cpu-a-clean @ 95bc0a6a357d** (expds/__init__, cpu_threads, lidar_a/{__init__,ground,terrain,tiling,runner,dtm_workers}) и `bot-8/bench/bench_acc_gpu.py` (sha256 ecb683db…, вне git). Манифест хранит sha256 по файлам. Есть assert «нет ehimen|nathan»; wt-cpua не используется.
  - Прогоны: stub_30M_1, unet_30M_1 (reps 1), stub_30M_d8, stub_75M_1, как в acc_gpu_T4.md. Плюс клоки nvidia-smi.
  - Результат: `/kaggle/working/bench_acc_gpu_t4.json` (host / device_exact / device: wall/cpu, byte_equal, max_abs, peak_vram_mb, x_wall) и `bench_acc_gpu_t4_clocks.csv`.
  - Ограничения: общий потолок `CNX_BENCH_MAX_S` = 1800 с (по умолчанию), каждый прогон — подпроцесс с таймаутом, запас 300 с до DEADLINE. Отключить: `CNX_BENCH=0`.
  - Любая ошибка записывается в JSON/лог и **не влияет на код выхода smoke**.
  - CPU-проверка (`tests/test_bench_wrapper.py` → `logs/test_bench_wrapper.json`): путь «нет CUDA» → skipped; обёртка из реального ядра с `--dev cpu` и малым px → 3 прогона rc 0, device_exact byte_equal True, max_abs 0.
- **Пересборка GPU и TPU.** build rc 0. Вендор и RELIEF_SHA не изменились (relief_sha 6ed689552b4a9780). Все 16 kernel-metadata.json побайтно совпадают с бэкапом `backup/cnx_contract_2340/`. py_compile: 26 файлов OK.
- **Новые sha256:**
  - src convnext_main.py 44603fd7…
  - smoke 508cb3d2…
  - prep 586d7257…
  - TPU smoke 57752e45…
  - (все — `sha256sum`, см. STATUS.md)
- **Запущенный prep (5033bbfb…) не затронут.** Полные ядра проверяют только RELIEF_SHA и теги, а они не изменились. Локальный prep.py пересобран из нового src (так требует TPU build). Его код отличается только добавленными шагами 9б/11, которые prep не исполняет.

## 29.10.2026 ~02:32 МСК — правки перед GPU: threshold_group и бюджет после прогрева (бот 9; команды пуша НЕ изменились)
Бэкап до правок: `backup/cnx_pre_gpu_0235/` (+ SHA256SUMS_before.txt). Изменён только `src/convnext_main.py` (sha 44603fd7… → 809a2f0d…) и `src/validate_oof.py`.
- **(1) `threshold_group="li_cnx"`** во всех детекциях OOF (`decode`, стр. 612–613; `markup_type="li"` оставлен). Других писателей детекций в ядре нет: тест-регионы фолда = OOF, объединённые `oof_cnx_{v}.geojson` и `merge_oof.py` копируют свойства. GT-файлы (`gt_li_cnx_*`) не менялись. `validate_oof.py`: ключ добавлен в KEYS и проверку (стр. 12, 19, 40).
- **(2) Бюджет итераций** по медиане с/ит итераций 101–150 (`CNX_BUDGET_WARM_IT`=100, `CNX_BUDGET_IT`=150; CFG стр. 105–106; цикл стр. 496, 512–517). Раньше было среднее за ит. 1–50 вместе с прогревом загрузчиков/cudnn. dt честный: `loss.item()` синхронизирует каждую итерацию. Лог: «с/ит: медиана ит. 101–150 …, среднее 1–150 …». При iters ≤ 150 бюджет не считается (smoke 10 ит. — как раньше).
- workers не трогал (2 на полосу × 2 полосы на 4 vCPU — возможное узкое место, у D-FINE две полосы давали +28 % с/ит).
- Проверки (OMP 2, nice 10): test_head / test_contract / test_bench_wrapper rc 0 (`kaggle/convnext_lidar/logs/*_0235.log`). Сухой прогон пересобранного smoke c `CNX_ITERS=8 CNX_BUDGET_WARM_IT=2 CNX_BUDGET_IT=5`: exit 0, 20,3 с, пик RSS 1 815 596 КБ; «с/ит: медиана ит. 3–5 0.325, среднее 1–5 0.390 → бюджет OK»; 1583 + 394 детекции, у всех `threshold_group=li_cnx`, `markup_type=li`; validate_oof «ошибки схемы: нет», идеальные круги 0.9999994444 / 0.9999991667; check_maps 388/388, 1583/1583 (`logs/dry_run_0235.log`, `dry_run_0235_rss.txt`, `validate_oof_0235.log`). Выход прогона удалён.
- Пересборка `build.py` (`CNX_BENCH_COMMIT=95bc0a6a…`): rc 0, вендор побайтно тот же (manifest = бэкап), RELIEF_SHA тот же (relief_sha 6ed689552b4a9780), бенч bot-8 тот же; 12 kernel-metadata.json не изменились; в каждом ядре только эти правки + новый gzip-заголовок VENDOR/BENCH_B64. sha256: `kaggle/convnext_lidar/logs/sha256_pre_gpu_0235.txt` (smoke 95d72749…). prep-ядро пересобрано тем же скриптом (его ветка кода не менялась); запущенный prep не затронут, перепушивать prep не нужно.
- TPU: `convnext_lidar_tpu/build.py` собирает из того же `convnext_main.py` → правка (1) попала автоматически. train_job в TPU заменён `tpu_train.py`, там бюджет уже по установившейся с/ит после WARM_IT. Пересобран **только koz3-cnx-li-tpu-sc** (53166c8e…). tpu-smoke (в очереди), tpu-c3sc, tpu-c5sc возвращены из бэкапа без изменений (smoke 57752e45…): у них threshold_group нет → проставить при сборке стека или пересобрать перед пушем.

## 29.10.2026 ~06:25 МСК — koz3-cnx-li-c3sc-f01 COMPLETE: разбор, FP f0, честная метрика (бот 9)
Вход: `/workspace/kaggle_out/koz3-cnx-li-c3sc-f01/` (только чтение). Копии лога/отчёта и вся работа лежат в `kaggle/convnext_lidar/c3sc_f01_out/` (`log_plain.txt`, `report_cnx_full.json`, `metric/`).

**1. Здоровье лога** (`log_plain.txt`, `report_cnx_full.json`)
- 2×T4. Обе полосы дошли до 12000/12000 итераций, бюджет **не урезан**:
  - f1 (cuda:1): «медиана ит. 101–150 0.322, среднее 1–150 0.520 → бюджет OK», прогноз 1,07 ч; t_train 3922,8 с.
  - f0 (cuda:0): 0.343 / 0.575, OK, прогноз 1,14 ч; t_train 4041,6 с.
  - Медианный прогноз совпал с фактом. Итог 4244 с (≈71 мин), «ИТОГ: все задания OK», errors [].
- Loss (средние по истории) снижается:
  - f1: 3,25 → 2,25 → 1,79 → 1,57 → 1,30 → 1,39 → 1,33;
  - f0: 5,43 → 2,55 → 2,07 → 1,81 → 1,51 → 1,44 → 1,39.
  - Единичные пики на батчах (f1 ит. 600 = 7,36) не страшны.
- **VRAM в логе и отчёте нет.** data_wait тоже не пишется.
- Узкое место CPU при двух полосах не видно. s/it 0,32–0,34 ниже оценки 0,37–0,45 для одной полосы (§7, это оценка, а не замер). Разница полос 6,5 %, а у D-FINE было +28 %.
- Предупреждения безвредны: requires_grad UserWarning (самотест головы, script.py:132), фолбэк Kaggle Secrets → приватный датасет, SyntaxWarning nbconvert/mistune.
- circle_r_m: строка CFG в логе печатается до переопределения из HEAD (стр. 136–137 ядра), поэтому там 1.0. Фактически 0,5 м: так в cfg отчёта, и радиус круга в OOF на земле = 0,4988 м. Это косметика.

**2. li-only баллы и FP в f0**
- Баллы из отчёта (порога нет, берутся все детекции ≥ min_score 0,05): f1 0,4703, f0 0,2168.
- f0 состоит из 6 участков Гнёздово, GT 838. f1 — 1 участок Тимерево, GT 387.
- Детекции на 1 GT: f0 11,5 (9604/838), f1 2,5. По ≥0,05 9604 действительно много, но это **хвост низкой уверенности**.
- Курганы f0 при пороге ≥0,3: 1003 детекции на 837 GT. Из них povr 347 TP / 222 FP, tselye 172 / 262 (ana.json cnx_thr_table).
- Почти всё лишнее — в 101_Днепровская и 102_Заольшанская:
  - при ≥0,05: 3472 и 3288 детекций на 115 и 68 GT li;
  - при ≥0,3: 155 и 79 (stats2.json).
- Лучший порог по классам (in-sample, только cnx):
  - Гнёздово: povr 0,329, F1 0,610; tselye 0,459, F1 0,568.
  - Тимерево: povr 0,064, F1 0,675; tselye F1 0,02 (9 GT).
- **Вердикт:** в основном это калибровка, а не «мусор». НО калибровка **разная на разных участках**. Тимерево недоуверен: при ≥0,3 povr даёт только 93 детекции на 378 GT. Гнёздово переуверен на хвосте. Порог, перенесённый с одного участка на другой, ломается (см. п. 3).
- Настоящие FP тоже есть:
  - (а) 2254 точки из 8079 выдают **оба класса курганов** (nms_cross_class=false);
  - (б) маски fort/gor: 199 fort и 46 gor с уверенностью 0,53–0,96 при 0 fort GT и 1 gor GT. Честные пороги их отсекают (fort 0,92–0,96).
- Стек на тех же участках (GT li, `sli` = объекты базы с markup li): in-sample 0,476. cnx один — 0,713, стек li + cnx — 0,685 (ana.json). In-sample cnx явно сильнее li-ветки стека. LOFO на 2 семьях: 0,187 / 0,236 / 0,204 (порог по одной семье — шумно).

**3. Честная метрика: стек + li_cnx** (`metric/`, бот-10 koz3-1.5.0, global)
- База затронутых юнитов совпадает с bot-10/work/curves побайтно. Затронуты 2 юнита из 100.
- Результаты:
  - X (база + 10 578 cnx): k10 Δ −0,0105 (drop) / −0,0112 (keep), P 0,18, LOFO −0,0108, seeds 0/5, heavy −0,0105 → П1–П4 ✗. Вклад классов: povr −0,0100, tselye −0,0006.
  - Xx (то же с подавлением дублей классов): Δ −0,0081, P 0,21, 0/5 ✗.
- Механизм (fam_delta.json):
  - когда Гнёздово в тесте, порог li_cnx/povr подобран **только по Тимерево** и равен 0,0507 → Гнёздово povr 613 TP / 5142 FP (у базы 372/678), балл Гнёздово 0,337 → 0,285;
  - Тимерево: X −0,0036, Xx +0,0039.
- То есть минус даёт перенос порога между двумя по-разному откалиброванными семьями, а не модель. Пока у li_cnx 2 семьи, честная проверка вырождена. Решение — после c3sc-f23 + c3-f4, когда li_cnx будет на всех 44 семьях. П5: с таким Δ в пакет тоже не брать.

**4. Рекомендации для c3-f4 / c5*** (НЕ пересобирал)
- (1) Писать `torch.cuda.max_memory_allocated()` по полосам и долю data_wait в лог и отчёт.
- (2) В отчёте, кроме балла без порога, печатать per-class лучший порог/F1 и балл при 0,3. Сейчас 0,2168 вводит в заблуждение.
- (3) Печатать CFG после переопределения CIRCLE_R.
- (4) min_score 0,05 НЕ поднимать: у Тимерево лучший порог povr 0,064, при пороге 0,1 TP падает 301 → 234.
- (5) Подавление дублей «один центр — два класса» делать в стеке (постобработка; Xx лучше X на +0,0024 k10), а не в ядре: в ядре лучше сохранить оба класса.
- (6) Калибровку между участками ядром дёшево не исправить. Нужны все фолды (пороги по многим семьям) или отдельная калибровка в стеке.
- Багов, ломающих результат, не нашёл.

## 29.10.2026 ~06:37 МСК — перед c3-f4: VRAM/ожидание данных, пороги в отчёте, CFG после CIRCLE_R (бот 9; команды пуша НЕ изменились)
Бэкап до правок: `backup/cnx_pre_c3f4/` (convnext_main.py 809a2f0d…, build.py, validate_oof.py, CONVNEXT/STATUS, `kernels/*` .py+metadata всех GPU/TPU-ядер, SHA256SUMS_before.txt). Изменён только `src/convnext_main.py` (809a2f0d… → **6f0087a8…**). Только лог/отчёт: обучение, инференс, формат выходов, threshold_group, бюджет, prep/RELIEF_SHA, шаг бенча bot-8, metadata — не тронуты.
- **(1) VRAM и доля ожидания данных по полосам.** `_vram(dev)` (стр. 478) → `max_memory_allocated_mb` / `max_memory_reserved_mb` (на CPU/XLA — null). В train_job: `reset_peak_memory_stats` в начале; data_wait = время от конца итерации до получения батча (без лишних sync — `loss.item()` и так синхронизирует). Лог «память/загрузчик (обучение): …», отчёт `jobs[].vram_train`, `data_wait_s`, `data_wait_share`, `data_wait_share_after_warm`; после инференса `jobs[].vram_total` (пик за всё задание).
- **(2) Пороги в отчёте.** Рядом с `score_li_only` (без порога) — `score_li_only_thr`: `final_score_thr0_3`, `per_class_thr0_3`, `best_per_class` (порог/F1/tp/fp/fn на сетке 0.05…0.95 по каждому классу с GT), `final_score_best_thr` (пометка: in-sample, оптимистично). Лог: «score(Li-only) при пороге 0.3: … | лучший порог по классам: … | итог при лучших: …». Лог-строки обёрнуты в try и читают через `.get` (в TPU-сборке нет vram_train/data_wait_*).
  - Цена: на реальном OOF c3sc-f01 на боксе (2 потока, nice 10) f0 9604 детекции — 222,7 с, f1 974 — 32,6 с (`kaggle/convnext_lidar/logs/score_thr_c3sc_f01_check.log`). На Kaggle это +~1–4 мин CPU на полосу после инференса, бюджет итераций не меняется.
  - Заодно по c3sc-f01: f0 при пороге 0.3 — 0.517, при лучших порогах по классам 0.737 (повр. 0.25/0.666, целые 0.35/0.646, городища 0.9/1.0); f1 — 0.225 / 0.470. In-sample, оптимистично (тот же лог).
- **(3) CFG** печатается после `CFG["circle_r_m"] = CIRCLE_R` (стр. 131–132); в сухом прогоне `circle_r_m: 0.5`.
- Проверки (OMP 2, nice 10): test_head / test_contract / test_bench_wrapper rc 0 (`kaggle/convnext_lidar/logs/*_c3f4.log`). Сухой прогон пересобранного smoke (`_scratch/dry_run.sh`, `CNX_ITERS=8 CNX_BUDGET_WARM_IT=2 CNX_BUDGET_IT=5`, плюс `HF_HUB_OFFLINE=1 ULIM=3500000`: при ulimit -v 2 ГБ hf_hub падал с MemoryError на загрузке весов ImageNet из кэша; RSS-сторож 1,95 ГБ тот же): exit 0, 18,2 с, пик RSS 1 906 928 КБ; новые поля есть в логе и `report_cnx_smoke.json` (VRAM null на CPU, data_wait_share 0.0134/0.0151, thr0.3 0.0, лучший 0.0373/0.0); 1583 + 394 детекции, все `threshold_group=li_cnx`, `markup_type=li`; validate_oof «ошибки схемы: нет», 0.0373394759 / 0.0, идеальные круги 0.9999994444 / 0.9999991667; check_maps 388/388, 1583/1583 (`logs/dry_run_c3f4.log`, `dry_run_c3f4_rss.txt`, `validate_oof_c3f4.log`). Выход прогона удалён.
- Пересборка `build.py` (`CNX_BENCH_COMMIT=95bc0a6a…`, `logs/build_pre_c3f4.log`), финальная в 06:30:01 МСК: rc 0. Во всех 12 ядрах отличие от бэкапа — только src-правки + gzip-заголовок VENDOR_B64 (и BENCH_B64 у smoke), распакованное содержимое побайтно то же → вендор, RELIEF_SHA (relief_sha 6ed689552b4a9780), бенч bot-8 те же; 12 kernel-metadata.json побайтно = бэкап. sha256: `kaggle/convnext_lidar/logs/sha256_pre_c3f4.txt`.
  - c3-f4 03c1366c… (по просьбе бота 1 держался в `hold/` до конца проверок, в 06:36 возвращён в `convnext_lidar/koz3-cnx-li-c3-f4`, в hold/ копии нет); c5sc-f01 7051fb89… (пересобран в 06:30:01, до 07:00); c5sc-f23 38f4b356…; c5-f4 a504a20a…; smoke 95e2872f…; prep 9e1e4397… (его ветка не менялась, перепушивать не нужно).
  - c3sc-f23 пересобран локально (fdb21d83…): это **не** запущенная версия (e3c44c81…, RUNNING) — её не перепушивать.
- TPU: `convnext_lidar_tpu/build.py` (`logs/build_pre_c3f4.log`), пересобран только **tpu-sc** (e785ab30…, test_tpu_parts CPU OK); tpu-smoke / tpu-c3sc / tpu-c5sc возвращены из бэкапа побайтно (57752e45… / 034dbf64… / 11e29b84…).

## 29.10.2026 ~08:20 МСК — c3sc-f23 COMPLETE — разбор (бот 9)
Вход `/workspace/kaggle_out/koz3-cnx-li-c3sc-f23/` (только чтение); работа — `kaggle/convnext_lidar/c3sc_f23_out/` (`log_plain.txt`, `report_cnx_full.json`, `metric/`, `metric_summary.txt`). Чекпойнты не копировались.
- **Прогон**: обе полосы 12000/12000 итераций, «ИТОГ: все задания OK за 4216.2 с», errors []. f2 (cuda:0) 0.2966 с/ит, обучение 3840.9 с, инференс 103.5 с; f3 (cuda:1) 0.2993 с/ит, 3854.2 с, 72.7 с. VRAM / data_wait / поля порогов в логе нет — ядро запущено до патча 06:30. Средний loss по блокам 2000 ит.: f2 2.53→1.26, f3 2.68→1.39 (пик 8.37 на ит. 1200).
- **OOF** (`metric/stats.json`): 39 424 полигона (f2 18 730, f3 20 694); все threshold_group = li_cnx, markup li, crs EPSG:3857, 0 невалидных. 3880 центров у 037_Никола_Ленивец вне бокса bot-10 — это не ошибка CRS: лидарный растр шире (по meta в maps/), а 8 GT Николы лежат в малом пятне, т. е. остальное — гарантированные FP.
- **li-only по фолду** (`metric/score_thr_f{2,3}.json`, compute_metrics koz3-1.5.0, gt_li_cnx фолда): f2 — без порога 0.1345, при 0.3 — 0.3296, при лучших порогах in-sample — 0.3955 (повр. 0.45/0.239, целые 0.25/0.523, городища 0.9/0.593, форт. 0.85/0.237). f3 — 0.0759 / 0.3470 / 0.3648 (повр. 0.35/0.362, целые 0.35/0.538).
- **Давление FP**: при 0.05 повр. f2 106 TP / 11 358 FP (GT 132), f3 102 / 12 306 (GT 129); при 0.3 — 50/542 и 56/282 (`metric/ana.json`, cnx_thr_table). Крупнейшие участки: Никола 9010 детекций, Боровичи_1 8643. Курганных центров с двумя классами сразу — 8671 из 30 106 (`stats2.txt`).
- **Честная метрика** (paired, k10, global; база A = 0.314615): X (+cnx f23) **−0.0115**, P 0.000, LOFO −0.0111, 0/5 сидов. Xx −0.0104. **f0–f3 вместе** X4 **−0.0114** (P 0.006, CI [−0.0237; −0.0022]), Xx4 −0.0112. П1–П4 ✗ везде, П5 — **не включать**. Сильнее всего страдает f2: −0.108 на участках фолда (повр. FP 193→664).
- **Пороги честные**: li_cnx подбирается на 9 из 10 k10-фолдов, т. е. leave-family-out (`fam_delta.json`). С f0–f3 подбор идёт по ~20+ семьям, а не по одной, как в f01. Порог повр. 0.30–0.43 при оптимуме по фолдам CNN 0.43–0.45 (in-sample). Дело не в вырожденном переносе порога, а в том, что cnx в виде добавки к стеку проигрывает.
- **Вывод**: c3sc как добавка (X/Xx) отклонён и на 4 фолдах. Следующий шаг — замена li-курганных групп (`replace_eval/`, раздел ниже).

## 29.10.2026 ~09:10 МСК — c3-f4 ERROR — fork/filelock (бот 9)
**Что случилось.** koz3-cnx-li-c3-f4: полоса `cuda:1 c3-sc:4` упала на 60-й с при старте воркеров DataLoader:
`RuntimeError: os.fork is unsafe while filelock is changing descriptor ownership` (filelock 4.2.x, `_api._audit_fork_safety`, Python 3.13).
Аудит-хук filelock на `os.fork` падает, если **другой поток** сейчас сам внутри fork (`fork_owner_depths`) или меняет владельца
дескриптора блокировки. Полосы — `threading.Thread`, у каждой DataLoader с `num_workers=2` (fork); в c3-f4 обе полосы стартовали
одновременно (c3-in ещё строил модель timm/HF с filelock и форкал своих воркеров) → гонка. Полоса `cuda:0 c3-in:4` доработала полностью.

**c3-in:4 валиден и сохранён** (не перезапускать, не перезаписывать): `/workspace/kaggle_out/koz3-cnx-li-c3-f4/`
— 12000 ит., n_det 2582, Li-only 0.2715 / thr0.3 0.3824 / лучший порог 0.4026; OOF `oof_cnx_c3-in_f4.geojson` (li_cnx, EPSG:3857).
`oof_cnx_c3-sc.geojson` там пустой — не использовать. Лёгкие копии + README для бота 10: `kaggle/convnext_lidar/c3in_f4_out/`.

**Фикс (`src/convnext_main.py` 6f0087a8… → 3f295186…; бэкап `backup/cnx_pre_forkfix_0905/`).**
- `FORK_LOCK = threading.RLock()` — один на процесс. Под ним: `build_model` (timm/HF-hub filelock) и новая `make_loader`
  (создание DataLoader + `iter` + 1-й батч, т. е. fork persistent-воркеров целиком). Соседняя полоса ждёт; `itertools.chain([first], it)` — 1-й батч не теряется.
- `_mp_ctx`: `CNX_MP_CTX=auto|spawn|fork|0`. **spawn по умолчанию не включается**: ядро — скрипт без main-guard, WinDS живёт в `__main__`;
  проверено на боксе (torch 2.15, Py 3.13) — spawn-воркер перезапускает весь скрипт как `__mp_main__` (в ядре: pip, S3, полосы…),
  а лямбда `worker_init_fn` не пиклится (PicklingError). Поэтому auto = spawn только если датасет не из `__main__` (сейчас нет) → иначе
  fork под FORK_LOCK. Если воркеры не стартовали — повтор с workers=0 (лог + `loader_fallback_error` в отчёте). Workers по умолчанию 2.
- `worker_init_fn` — функция `_worker_init` вместо лямбды. В отчёт задания: `loader_workers, loader_ctx, loader_fallback_error, loader_lock_wait_s, loader_start_s`.
- `CNX_SHARING` (опц., напр. `file_system`) → `torch.multiprocessing.set_sharing_strategy`; по умолчанию не меняется (к причине не относится).
  Фикс-аудит filelock не отключался.
- Проверка: `tests/test_forklock.py` (2 потока × 50 загрузчиков, fork, 2 воркера + поток с filelock): без замка 22/100 стартов упали
  ровно этой ошибкой (фолбэк на 0 их спас), с FORK_LOCK 0/100 (`logs/test_forklock_0905.log`). Сухой прогон smoke на новом коде (CPU,
  `CNX_ITERS=8`, HF offline, ULIM 3,5 ГБ): exit 0, 44,5 с, пик RSS 1 938 116 КБ; 1583 + 394 детекции, 0.0373394759 / 0.0 — как в прогоне 06:37
  (`logs/dry_run_forkfix_0905.log`). На CPU воркеров нет — путь fork проверен только тестом выше.
- TPU: `tpu_train.py` заменяет train_job целиком (свой DataLoader), полосы TPU — процессы, не потоки → не трогал. Якоря `convnext_lidar_tpu/build.py`
  на новом src применяются (проверено), но его проверка «prep собран из того же convnext_main.py» теперь не пройдёт, пока не пересобран prep
  (prep-ядро перепушивать НЕ нужно — его выход тот же).

**Сборка (`build.py`, e953558d… → 06e372d7…).** `c3-f4` убран из FULL → `DONE = {"c3-f4": "c3-in:4"}` (больше не собирается); новое ядро
`c3sc-f4` = `c3-sc:4` (одно задание); проверка «20 пар ровно один раз» по FULL+DONE. `CNX_BUILD_ONLY=<slug,…>` — собрать только указанные.
Пересобраны только (`CNX_BUILD_ONLY=koz3-cnx-li-c3sc-f4,koz3-cnx-li-c3in-f01,koz3-cnx-li-c3in-f23`, `logs/build_forkfix_0905.log`, `logs/sha256_forkfix_0905.txt`):
| ядро | .py sha256 | metadata sha256 |
|---|---|---|
| koz3-cnx-li-c3sc-f4 (новое) | 71ad2bf2d5113a79ef93df5d1f53b728d3ec4097f0e93564577398261d27151d | 0bdc582dc94b4652906a0170e8f231738ebf09a274f95dd6a4779cb5ab995024 |
| koz3-cnx-li-c3in-f01 | 848893a53b5fa94ad7349921f829cea5ee1ba3a41c3b6ca44e0ef98e5d0e9a2c (было db371496…) | fac9de8f… (без изменений) |
| koz3-cnx-li-c3in-f23 | 9a54a17bfe94f19ebfb8898b7a98edeb84694aed0b810b9f4c31d952fd185a53 (было 09813454…) | cea408be… (без изменений) |
Вендор (распакованный tar) и FOLDS — байт-в-байт как в прежних ядрах (отличается только mtime в gzip-заголовке VENDOR_B64). metadata c3sc-f4 = c3sc-f23 кроме id/title/code_file.
Не пересобраны (старый код, дуальные полосы): c3sc-f01/f23 (отработали), c5in-f01/f23, hold/c5-f4, hold/c5sc-f01, hold/c5sc-f23, smoke, prep, koz3-cnx-li-c3-f4 (оставлен как был, не пушить).

Пуш (бот 1):
```
cd /workspace/koz3_shared/bot-9/kaggle/convnext_lidar
KAGGLE_API_TOKEN=$(cat ~/.kaggle/access_token) ~/.local/bin/kaggle kernels push -p koz3-cnx-li-c3sc-f4
KAGGLE_API_TOKEN=$(cat ~/.kaggle/access_token) ~/.local/bin/kaggle kernels push -p koz3-cnx-li-c3in-f01   # ImageNet — только замер
KAGGLE_API_TOKEN=$(cat ~/.kaggle/access_token) ~/.local/bin/kaggle kernels push -p koz3-cnx-li-c3in-f23   # ImageNet — только замер
```
