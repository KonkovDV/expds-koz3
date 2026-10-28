# D-FINE-L на SpOr (детектор рамок): ядро Kaggle (бот 9)

Статус на 28.10.2026 22:05 MSK: падение smoke v1 (meta-тензоры) **исправлено**, тренер бота 5 встроен, ядра **пересобраны, НЕ запушены** (пушит бот 1). В сдачу результат пойдёт только после письма организаторов о весах.

## Цель и рецепт
- Источник рецепта: `/workspace/koz3_shared/bot-5/NOLIDAR.md` §6.1 (и §4.1). Там же оценка прироста +0.015 (0.007–0.029) и критерий приёмки Δ honest A ≥ +0.005. Я эти цифры не измерял.
- Реализация: 0,5 м/пикс (WarpedVRT в UTM из UTM.json: average при уменьшении, иначе bilinear), окно 640 px (320 м), 5 классов, 5 семейных фолдов из `folds_family_k5.json`, как в верификаторе. Ассерт: одна семья не может попасть в два фолда.
- Обучение:

| Параметр | Значение |
|---|---|
| Итерации / батч | 8000 ит, bs 8 |
| Оптимизатор | AdamW lr 1e-4, backbone ×0.1, wd 1e-4 |
| Расписание | OneCycle pct 0.05, clip 0.1, fp16 |
| Сэмплинг | 60 % окон с объектами; вес класса 1/√n с потолком 4×; сдвиг ±1/3 окна; рамка остаётся, если ≥50 % внутри окна |
| Аугментации | blur p0.3; цвет; gamma p0.3; gray p0.15; hflip + rot90; QAUG p0.5 (downscale 1.5–3.2, JPEG 15–60, шум) |
| Если loss NaN | `DFINE_NUM_DN=0` |

  Код EhimenNathan не копировался.
- OOF-инференс по полному растру каждого тестового участка фолда:
  - скользящее окно 640, перекрытие 160;
  - flip-TTA, ≤100 детекций на окно, score ≥ 0.05, NMS 0.5 по классам на tif;
  - рамка превращается во вписанный 32-угольный эллипс в EPSG:3857.

## Веса и лицензия (отметка для сдачи)
- Веса HF `ustc-community/dfine-large-coco`, ревизия закреплена: `aa8f8211545e4cff5adf52ea512f9ac74a9824ec`. Лицензия Apache-2.0, предобучение на COCO, модель D-FINE (Peng et al., 2024), 31,36 млн параметров (с 5 классами — 31,09 млн, `check_model` в `dryrun_fix_*/dfine_report.json`).
- Веса скачиваются из интернета в ядре (snapshot_download) и проверяются в самом начале.
- **Не сдавать, пока организаторы не ответят письмом про использование этих весов.**

## Папки и сборка
- Исходник: `kaggle/dfine_spor/src/dfine_main.py`. Сборка: `python3 kaggle/dfine_spor/build.py` создаёт:
  - `kaggle/dfine_spor/smoke/` — `dmitrijkv/koz3-dfine-spor-smoke`, `DFINE_MODE_DEFAULT='smoke'`. 10 итераций, 8 самых маленьких обучающих участков, ≤2 фолда (по числу GPU), по 1 тестовому участку на фолд. Печатает с/ит и «ПРОГНОЗ full».
  - `kaggle/dfine_spor/full/` — `dmitrijkv/koz3-dfine-spor-full`, 8000 ит × 5 фолдов. Веса каждого фолда сохраняются в `/kaggle/working/dfine_f{k}/`.
- Метаданные обоих ядер: private, GPU T4, internet on, `dataset_sources: ["dmitrijkv/koz3-s3-cred"]`. docker_image взят из `/workspace/kpull/s4/kernel-metadata.json`.
- Команды push (выполняет бот 1; сначала smoke, full — только после успешного smoke). Одновременно — только одно GPU-ядро: второе молча получает CPU.
  ```
  python3 /workspace/koz3_shared/bot-9/kaggle/dfine_spor/build.py
  KAGGLE_API_TOKEN=$(cat ~/.kaggle/access_token) ~/.local/bin/kaggle kernels push -p /workspace/koz3_shared/bot-9/kaggle/dfine_spor/smoke
  KAGGLE_API_TOKEN=$(cat ~/.kaggle/access_token) ~/.local/bin/kaggle kernels push -p /workspace/koz3_shared/bot-9/kaggle/dfine_spor/full
  ```

## Секреты
Ключи ищутся по порядку:
1. Kaggle Secrets `KOZ3_S3_ACCESS_KEY_ID`, `KOZ3_S3_SECRET_ACCESS_KEY`.
2. env `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY`.
3. Запасной путь: `/kaggle/input/**/cred.json` (поля `id`, `key`) из приватного датасета `dmitrijkv/koz3-s3-cred`, как в версии верификатора у бота 1.

Ключи не логируются и не пишутся в `/kaggle/working`. В лог попадает только тип ошибки (пример: `dryrun/dryrun.log`, строка «Kaggle Secrets недоступны: ModuleNotFoundError»).

## Фикс падения Kaggle smoke v1 (28.10, «Cannot copy out of meta tensor»)
**Что было.** `/workspace/kaggle_out/koz3-dfine-spor-smoke/dfine_report.json`: фолды 2 и 3 упали на `make_model().to(dev)` (script.py стр. 422), всего 51 с. В `koz3-dfine-spor-smoke.log` стр. 40–43 и 56–59 `class_embed.*` и `bbox_embed.*` помечены MISSING. Это верхние головы, которые в D-FINE связаны с `model.decoder.*` через `_tied_weights_keys` (`transformers/models/d_fine/modeling_d_fine.py` стр. ~1954).

**Версии на Kaggle** (`koz3-dfine-spor-smoke.log`):
- стр. 1: `pip install: ['transformers==5.21.0']`, то есть в образе была другая версия и сработал наш пин;
- стр. 4–10: torch 2.11.0+cu128, torchvision 0.26.0+cu128, transformers 5.21.0, rasterio 1.5.1, boto3 1.41.5;
- стр. 14: 2× Tesla T4;
- версия huggingface_hub не логировалась. Теперь ядро её печатает.

**Корневая причина — гонка потоков, а не версия.**
- `from_pretrained` в transformers 5.21.0 входит в контекст `init.no_tie_weights()` (`transformers/initialization.py` ~стр. 291).
- Этот контекст на время загрузки подменяет `PreTrainedModel.tie_weights` на пустышку, на уровне класса, то есть для всего процесса, и восстанавливает в `finally`.
- Ядро грузит модели двух фолдов одновременно в двух потоках (`gpu_worker` на 2×T4). Второй поток запоминает уже подменённую пустышку как «оригинал». В результате пост-загрузочный `model.tie_weights(...)` (`modeling_utils.py` стр. ~4535) ничего не делает, и 48 тензоров голов остаются на meta.
- Одиночная загрузка проходит. Поэтому старый box-дриран (`dryrun/dryrun.log`, 1 CPU-устройство) ошибку не поймал.

**Репро на CPU** (venv `kaggle/dfine_spor/fix/venv_t211`: torch 2.11.0+cpu, torchvision 0.26.0+cpu, transformers 5.21.0, huggingface_hub 2.3.0, safetensors 0.8.0; `fix/pip_t211.log` rc=0; веса — локальная копия HF-кеша в `fix/hf`, ревизия aa8f8211):
- `fix/repro_meta_t211.log` (`python repro_meta.py single|threads`):
  - single: meta 0, `to(cpu) OK`;
  - threads: 1 из 6 прогонов упал в обоих потоках с meta 48 (`class_embed.0.weight`…), `tied: False`, `NotImplementedError('Cannot copy out of meta tensor…')`.
- `fix/repro_meta_t215.log` (`/tmp/dfine_venv`, torch 2.15.0+cpu, transformers 5.21.0): single OK; threads — 1 из 3 упал так же.
- Вывод: падение на CPU воспроизводится, но недетерминированно. Зависит от гонки, а не от версии torch.

**Исправление** (`src/dfine_main.py`, блок между `# >>> MODEL` и `# <<< MODEL`; дифф — `fix/src.diff`; бэкап исходников — `backup/20261028_2152/`):
1. `MODEL_LOCK`: загрузки моделей идут строго по одной. Оригинальные `PreTrainedModel.tie_weights/init_weights` запоминаются в главном потоке. Если перед загрузкой они подменены, ядро их восстанавливает и пишет в лог «ВНИМАНИЕ: … восстанавливаю».
2. `_repair_meta`: если после загрузки на meta что-то осталось, вызывается оригинальный `tie_weights()`. Если и это не помогло: `to_empty(cpu, recurse=False)` + `_init_weights` по модулям, затем повторное связывание.
3. `check_model` — жёсткая проверка до `.to(dev)`:
   - на meta нет ни параметров, ни буферов;
   - `num_labels == 5`;
   - `decoder.class_embed[i]` и `enc_score_head` имеют форму (5, d_model);
   - `denoising_class_embed` содержит 6 строк;
   - верхние `class_embed/bbox_embed` делят память с `decoder.*`.
   При нарушении — понятный RuntimeError.
4. Пин: `_deps` ставит transformers==5.21.0 и после импорта **ассертит** `transformers.__version__ == "5.21.0"`, печатает huggingface_hub и safetensors.
5. Ранний чек «Шаг 2б» — до фолдов и чипов: `make_model(seed=0)` + `check_model` + CPU-forward на 1×3×128×128, ассерт logits[...,5] и pred_boxes[...,4]. Если не прошёл — `SystemExit("модель D-FINE не прошла ранний чек")` в первые минуты.

**Проверка фикса** (venv_t211 = версии Kaggle):
- `fix/test_model_block_t211.log` (`python fix/test_model_block.py`): `ALL_OK`. В том числе:
  - 2×2 одновременные загрузки в потоках через `make_model` — все OK;
  - «залипшая» подмена `tie_weights` восстанавливается;
  - модель, загруженная при подмене (meta 48), отвергается `check_model`;
  - после `_repair_meta` meta 0, forward даёт logits [1, 300, 5].
- `fix/stress_lock_t211.log` (`python fix/stress_lock.py 10`): `STRESS runs=10 loads_ok=20 loads_fail=0`.

## Тренер бота 5 (вендорен в ядро)
- Источник: `/workspace/koz3_shared/bot-5/repo`, ветка bot5-dfine-kaggle, коммит 0256251c8ee2de0fd03ca009e8ffed8463d198cc. Репозиторий бота 5 не трогал: файлы извлечены `git show` в `kaggle/dfine_spor/fix/bot5_0256251/`.

| Файл бота 5 | sha256 | git blob |
|---|---|---|
| scripts/det_train_hf.py | 78cab4ac41081c2dcfb3dbbe97492b4a42ccd309f49e934a8e7516a10b3f7f15 | 45a67faee1823acde38f930e5c6f3a7b6e3363b3 |
| dfine_folds.sh | dbf27ee5fb4a1be945191b3c6d1f916a004691c35d97321c7fbf1705bdc9b8c5 | 1890e35b… |
| README_KAGGLE.md | dca0bce8b1dda163bf8fa7a8854778255aaba68e6ca9c9d3394eb552634cd13f | bd1bc1fd… |

- **Адаптер** `src/dfine_train_bot5.py`, sha256 e6e388b013d2fb55047a26536f14673c3c55054782a3decc38b08682a1c29a36 (`sha256sum`, также печатается `build.py`):
  - `jitter()` — дословно из det_train_hf.py стр. 49–80;
  - `train(model, batches, cfg, device, log)` повторяет цикл бота 5 (стр. 266–282): AdamW с 2 группами (backbone lr×0.1), OneCycleLR(total_steps=iters, pct_start), GradScaler + autocast fp16 на cuda, clip_grad_norm, шаг расписания;
  - добавлено адаптером: стоп по бюджету фолда и дедлайну (`budget_s`, `stop_ts` = дедлайн − 30 мин), пропуск нефинитного loss (≤50 подряд), лог каждого шага в smoke и каждые 100 ит в full, `model.train_report` (trainer "bot5@0256251", iters, s_it_median, loss_first/last);
  - данные, фолды, OOF, ключи и ранние чеки остаются наши. Формат батча совпадает с выходом процессора бота 5: `RTDetrImageProcessor` — do_rescale, do_normalize=false; pixel_values 0..1; labels class_labels + boxes cxcywh норм. Поэтому `AutoImageProcessor` и COCO-датасет бота 5 не нужны.
- **Встраивание.** Kaggle загружает только файл кода, поэтому `build.py` вкладывает адаптер в скрипт строкой `BOT5_TRAINER_SRC`. Ядро берёт `dfine_train_bot5.py` рядом со скриптом или в `/kaggle/input/**`, иначе встроенную копию. В лог пишется «тренер: … | бот 5: … sha256».
- **Флаги** (env; на Kaggle действуют значения по умолчанию, для смены — поменять дефолт в `src/dfine_main.py` стр. 59–60 и пересобрать):
  - `DFINE_TRAINER=bot5` (по умолчанию) | `builtin` — запасной встроенный `train_loop` §6.1;
  - `DFINE_AUG=ours` (по умолчанию, с QAUG) | `bot5` — `jitter()` бота 5.
- **Загрузка модели у бота 5:** `AutoModelForObjectDetection.from_pretrained(ckpt, num_labels, id2label, label2id, ignore_mismatched_sizes=True[, num_denoising])` (det_train_hf.py стр. 139), без device_map и tie_weights. Пин в ноутбуке: `pip install -U "transformers>=4.57"` (`git show 0256251:kaggle/dfine_kaggle.ipynb`), то есть без фиксации версии. Бага у него нет только потому, что каждый фолд — отдельный процесс (dfine_folds.sh, полосы CUDA_VISIBLE_DEVICES). Аргументы загрузки у нас те же, поэтому оставлен наш lock + check + строгий пин.
- **Рецепт бот 5 vs наш:**
  - совпадает (det_train_hf.py стр. 192–204): lr 1e-4, backbone ×0.1, wd 1e-4, OneCycle pct 0.05, clip 0.1, bs 8, 8000 ит, 640 px, fp16 на T4, 60 % окон с объектами, вес класса 1/√n с потолком 4;
  - различия:

| | Бот 5 | Наш |
|---|---|---|
| Сэмплинг | по картинкам с возвращением (стр. 275) | наш поток окон со сдвигом ±1/3 |
| Аугментации | hflip 0.5, rot90, яркость ×U(0.75,1.25)+U(−20,20) p0.8 (без поканальной), gamma 0.3, gray 0.1, blur p0.2 σ0.3–1.5; strong (downscale 0.4–0.8 p0.3) — только по `--strong-aug`, в dfine_folds.sh не включён | blur p0.3 σ0.5–1.5, поканальный цвет, gray 0.15, QAUG p0.5 (downscale 1.5–3.2, JPEG 15–60, шум) |
| Smoke | bs min(bs,2), 10 ит | 10 ит, bs как задан |
| NaN-защита / бюджет времени | нет | есть (в адаптере тоже) |
| Ресайз | процессор | нативные окна 640 |
| Предсказание | сеточные PNG | полные растры |

  - заявление бота 1 «CPU-smoke бота 5: 12,19 с/ит, пик 1577 МБ» подтверждается файлом `fix/bot5_0256251/README_KAGGLE.md` стр. 33 (256 px, bs 2). Сам я это не перемерял.

## Проверки (box, CPU, ≤2 потока, nice 10)
- **Дрираны после фикса** — собранный `smoke/koz3-dfine-spor-smoke.py --allow-cpu`, скрипт `fix/run_dryrun.sh`:
  - условия: OMP 2, nice 10, сторож RSS 2000 МБ; ключи из env, в лог не попадают; 2 участка (161_Курганы_32, 051_Воронеж_2), tile 256, bs 1, 3 ит, 6 окон;
  - результаты (из `dryrun.log` и `dfine_report.json` в каждой папке):

| Папка `kaggle/dfine_spor/…` | torch / transformers | Тренер / аугм. | Ранний чек | с/ит (медиана) | Детекций | Пик RSS | rc |
|---|---|---|---|---|---|---|---|
| dryrun_fix_t211_bot5 | 2.11.0 / 5.21.0 | bot5 / ours | OK, logits [1,300,5] | 9.44 | 460 | 1385 МБ | 0 |
| dryrun_fix_t211_builtin | 2.11.0 / 5.21.0 | builtin / ours | OK | 6.45 | 476 | 1497 МБ | 0 |
| dryrun_fix_t211_bot5aug | 2.11.0 / 5.21.0 | bot5 / bot5 | OK | 6.39 | 471 | 1442 МБ | 0 |
| dryrun_fix_t215_bot5 | 2.15.0 / 5.21.0 | bot5 / ours | OK | 8.80 | 449 | 1481 МБ | 0 |

  - в каждой папке есть `oof_dfine_spor.geojson` и `oof_dfine_f1.geojson`. Например, 460 объектов, crs EPSG:3857, свойства class_name, confidence, crs, fid, fold, group, markup_type, original_crs, region_name, tif;
  - ошибок фолдов нет, `model_check` в отчёте: params 31.09 млн, heads [6, 5, 256];
  - пин = версия Kaggle (5.21.0), поэтому отдельного прогона «с пином» не требуется;
  - 2 GPU на CPU не эмулируются: CPU-путь — одно устройство. Многопоточность проверена тестами выше.
- **Утечка ключей:** `rg -F -f <значения AWS_* из ~/.koz3s3/env>` по всему bot-9 (кроме venv/hf/s3cache), включая smoke/, full/, src/ и dryrun_fix_*, даёт 0 файлов.
- **Сборка:** `python3 kaggle/dfine_spor/build.py`: smoke 64196 байт (sha256 7e98743d…), full 64195 байт (b23dbc06…), py_compile OK. Блок ключей бота 1 (`_cred_from_inputs` … `SystemExit("нет ключей S3…")`) побайтно совпадает с бэкапом.
- Ниже — старые проверки до фикса:
- **pip в чистом venv** (`/tmp/dfine_venv`, python3.13), `kaggle/dfine_spor/pip_clean.log`, rc=0:
  - transformers==5.21.0, rasterio==1.4.3, pyproj==3.7.1, shapely==2.1.1, boto3==1.40.61, opencv-python-headless==4.11.0.86, scipy==1.16.3 (`--only-binary :all:`);
  - torch 2.15.0 / torchvision 0.29.2 (CPU) — только для box. В ядре torch не ставится, используется образ Kaggle.
  - Ядро проверяет версии и при несовпадении ставит transformers==5.21.0. Затем жёсткий импорт `DFineForObjectDetection` — ранний фейл.
- **Ранние фейлы**: без GPU ядро падает (`--require-gpu` по умолчанию; `--allow-cpu` только для box). Дальше по порядку проверяются S3 list, скачивание весов HF и утечки семей в фолдах — всё до построения чипов.
- **Мини-прогон на CPU** (`kaggle/dfine_spor/dryrun/dryrun.log`):
  - условия: 2 участка (161_Курганы_32, 051_Воронеж_2), tile 256, bs 1, 3 итерации, 6 окон инференса;
  - чипов 21 (из них с объектами 13);
  - loss конечный: 415.7 → 343.6 → 377.5;
  - 446 детекций в `dryrun/oof_dfine_spor.geojson` (crs EPSG:3857; свойства class_name, confidence, region_name, crs, markup_type sp_or, original_crs, group sp_dfine, fold, tif, fid);
  - maxrss 1269 МБ, ошибок фолдов нет.
- Найден и исправлен баг: WarpedVRT не умеет boundless-чтение. Теперь пересечение окна с растром читается вручную, остальное дополняется нулями.
- Отдельный тест (команда в логе сессии): loss D-FINE с пустыми таргетами конечный (батч смешанный 656.3, только пустые 578.8), backward проходит, `num_denoising` по умолчанию = 100.
- Математика hflip/rot90 для рамок проверена аналитически: np.rot90 переводит (x, y) в (y, T−x).

## Выход для scoreboard бота 10
- `/kaggle/working/oof_dfine_spor.geojson` (все фолды) и `oof_dfine_f{k}.geojson`: полигоны EPSG:3857 в формате стека, group `sp_dfine`.
- `dfine_report.json`: cfg, ревизия весов, по фолдам iters, s_it_median, история loss, n_det, время инференса, `projection_full_h`.
- Дальше по §6.1 бот 10: `coco_preds_to_stack --radius 4 --merge-m 4`, группа sp_dfine, затем `scoreboard --replace/--add --honest`.

## Smoke v2 на Kaggle (запуск бота 1 в 22:07 МСК, COMPLETE): разбор
Источник: `kaggle/dfine_spor/smoke_v2_out/` (совпадает по `cmp` с `/workspace/kaggle_out/koz3-dfine-spor-smoke-v2/`): `koz3-dfine-spor-smoke.log`, `dfine_report.json`, `oof_dfine_*.geojson`.
- Все шаги пройдены. Импорт-чек, S3 [2 с], веса HF [23 с], «Шаг 2б: модель OK (CPU)» [25 с], «фолды OK», 132 участка SpOr. Фолды f2 (gpu0) и f3 (gpu1) завершились, `ошибки фолдов: []`, 1784 детекции, `total_sec` 70.
- OOF geojson валиден: EPSG:3857, Polygon, conf 0,30–0,75. Классы почти случайные — после 10 итераций это ожидаемо.
- Все loss конечны. f2: 687,8 → 813,5; f3: 72,1 → 79,1.
- Скорость на T4, оба GPU параллельно: итерация 1 ≈ 12,5 с, итерация 2 ≈ 6,7 с, дальше 0,91–0,95 с/ит. `s_it_median`: f2 0,937, f3 0,903. Принимаю **0,92 с/ит** (bs 8, 640 px, тренер бота 5).
- Пиковая VRAM в v2 не логировалась. Теперь пишется `peak_vram_gb` (по фолду, `torch.cuda.max_memory_allocated`) в `dfine_report.json`.
- Время по фазам (лог smoke v2): pip/импорт ≈ 42 с, веса ≈ 21 с, чипы 8 с (8 маленьких участков), обучение 22 с на фолд (10 ит.). Инференс: f2 — 9 окон за 7–8 с, f3 — 6 окон за 3 с.
- Строка «ПРОГНОЗ full … ≈ 5.11 ч» считает 8000·5/2 без учёта 3 «волн» и инференса — это занижение. Верный расчёт ниже.

### Проверки по флагам бота 1 (22:20)
- **10 train-чипов против 62 — артефакт smoke.** 6 из 8 самых маленьких участков попали в фолд 2, поэтому f2 учится только на чипах фолдов 3 и 4 (`fix/smoke_v2_check/chips640.log`, «CHK chips per fold»).
- **Но нашёлся настоящий баг в построении чипов** (`fix/smoke_v2_check/chips640.log`). Окно BIG ≈ 1067 px всегда центрировалось на объекте, а чип с >50 % nodata отбрасывался. На маленьких растрах и у краёв объекты терялись: 091_Сакские_курганы (761×675 px) — из 27 GT 0 чипов; 153_Курганы_24 (840×656) — из 20 GT 0; 034_Усть_река — 0 из 2. Итого из 89 GT получилось только 38 объектных чипов. Кроме того, фон выбирался через `hash(region)`, который солится в каждом процессе: число чипов было недетерминированным (65 на Kaggle против 59 локально).
  - Фикс в `src/dfine_main.py`:
    - окно прижимается к границе растра (`_clamp0`); если растр меньше окна — центрируется;
    - объектный чип отбрасывается, только если в рамке объекта <50 % валидных пикселей;
    - фон берётся через `random.Random(zlib.crc32(region))`;
    - кроп T×T в ChipSet центрируется на объекте (`c`) с jitter ±shift·T;
    - в лог пишется число отброшенных чипов.
  - После фикса (`fix/smoke_v2_check/chips640_fix.log`): 151 чип (89 объектных), сохранены все 89 из 89 GT. Пустых выборок 4/400 (ours) и 5/400 (bot5), до фикса было 75/83.
- **Формат целей проверен** (`CHK targets`, оба aug, по 400 выборок): рамки cxcywh в [0,1], нулевых w/h нет, рамок <4 px нет, метки 0..4, `bad: []`. Пустые цели дают конечный loss: адаптер бота 5 пропускает шаги с неконечным loss, а пустых целей теперь ≈1 %.
- **Рост loss — шум, а не расхождение.** На 10 итерациях OneCycle (pct 0,05) ещё в разгоне. На f2 почти пустые цели дают VFL ~580 (риск 3). CPU-тренд на 32 итерации (`fix/smoke_v2_check/train256_f3.log`: фолд 3, 256 px, bs 2, тренер бота 5, 2 потока, peak RSS 1,88 ГБ):
  - loss на фиксированном батче 108,55 → 60,59;
  - средний loss первой/последней четверти 150,3 → 77,7;
  - все значения конечны.

## Бюджет и выбор фолдов (решение 22:26: квота ~30 ч/нед., из них ~12 ч на ConvNeXt; потолок ~5 ч)
- Инвентарь фолдов — `fix/fold_inventory.py` → `fix/fold_inventory.{json,log}` и `fix/inv_sizes.py`. Окна инференса считались при 640/160, 0,5 м, UTM по заголовкам растров.

| фолд | участков SpOr | семей | GT-рамок | окон инференса | ГБ tif |
|---|---|---|---|---|---|
| 0 | 2 | 1 | 83 | 104 | 0,17 |
| 1 | 22 | 12 | 789 | 704 | 0,41 |
| 2 | 49 | 15 | 779 | 1108 | 2,45 |
| 3 | 16 | 12 | 290 | 483 | 1,45 |
| 4 | 43 | 16 | 1164 | 944 | 0,92 |

- 5 фолдов при 8000 ит. — 3 волны по ≈ 2,04 ч обучения + ≈ 0,2 ч инференса, итого ≈ 7 ч. Это больше 5 ч, поэтому работа разбита на два ядра.
- **`koz3-dfine-spor-full` (эта неделя): фолды 4, 2, 1.** Покрывают 2732 из 3105 GT (88 %), 43 из 56 семей и 114 из 132 участков. Это максимум покрытия для 3 фолдов: альтернатива 4, 2, 3 даёт 2233 GT.
  - Очередь `4,2,1`: gpu0 берёт f4, gpu1 — f2, f1 уходит первому освободившемуся GPU (f4 короче по инференсу).
  - `DEADLINE_H_DEFAULT = 5.0`, `FULL_ITERS_DEFAULT = 6000`.
- **`koz3-dfine-spor-full-rest` (следующая неделя): фолды 0, 3** — одна волна параллельно. `DEADLINE_H_DEFAULT = 3.5`, 6000 ит. Число итераций то же, чтобы OOF были сопоставимы.
- **Итерации на фолд: 6000** (OneCycle проходит полностью). При 0,92 с/ит это 5520 с ≈ 1,53 ч обучения на фолд.
- 8000 ит. не влезают. Расчёт для gpu0: 0,25 + 2,04 + 0,22 + 2,04 + 0,17 ≈ 4,7 ч, а стоп обучения стоит на 4,5 ч — f1 обрезался бы.
- Прогноз full (оценка):
  - старт ≈ 0,25 ч: pip 42 с, веса 21 с, скачивание 5,4 ГБ и чипы всех 132 участков. Время чипов на полном наборе не замерено — это главная неопределённость;
  - gpu0: f4 1,53 + ~0,2 ч инференса (944 окна при ≤0,85 с/окно по smoke), затем f1 1,53 + ~0,17 ч;
  - итого **≈ 3,7 ч стены** (gpu1 освобождается на ≈ 2,05 ч).
- Запас до потолка — ограничители в коде:
  - бюджет фолда = (DEADLINE − now − 1800)/фолдов_на_GPU − 1200. Для f4 на старте ≈ 7450 с, для f1 на ≈ 2,0 ч — ≈ 7800 с, при нужных 5520 с. Значит, с/ит может вырасти до ~1,3 до того, как адаптер обрежет обучение («[bot5] СТОП по времени»);
  - новое: если дедлайн уже прошёл, инференс оставшихся участков пропускается (`infer_skipped` в отчёте). Kaggle-лимит 12 ч не достигается.
- Прогноз full_rest ≈ 0,25 + 1,53 + 0,1 ≈ **1,9 ч**. Итого квота D-FINE за 2 недели ≈ 3,7 + 1,9 ≈ 5,6 ч.
- **Честный LOFO.** Утечку семей ловит ассерт в коде. OOF этой недели — только по фолдам 4, 2, 1 (114 участков). Фолды 0 и 3 (18 участков, 373 GT) в `oof_dfine_spor.geojson` ядра full отсутствуют, и в scoreboard их нельзя считать нулём/FN по D-FINE. Метрику стека сравнивать только на участках фолдов 4, 2, 1 — до объединения с OOF ядра full-rest.

## Проверки после фикса чипов и разбиения (CPU, 2 потока, nice 10, сторож 2 ГБ)
- `python3 kaggle/dfine_spor/build.py`: smoke 66656 Б (sha256 32ed4dd99313509a…), full 66730 Б (177cc263bdab1e64…), full_rest 66728 Б (c396346c9d48692b…). Все три проходят `py_compile`. Метаданные: id `dmitrijkv/koz3-dfine-spor-{smoke,full,full-rest}`, приватные, T4, интернет, датасет `dmitrijkv/koz3-s3-cred`.
- Сухой прогон smoke (`fix/dryrun_smoke3_t211/dryrun.log`): rc=0, 426 детекций, OOF EPSG:3857, peak RSS 1,50 ГБ.
- Сухой прогон full (`fix/dryrun_full3_t211/dryrun.log`): 2 участка, 3 ит. Порядок фолдов 4 → 2 → 1, бюджеты считаются от 5 ч, rc=0, 1407 детекций, peak RSS 1,52 ГБ.
- Grep значений ключей из `~/.koz3s3/env` по трём сборкам и всем новым логам/JSON — 0 совпадений.

## Пуш (делает бот 1, бот 9 не пушит)
```
python3 /workspace/koz3_shared/bot-9/kaggle/dfine_spor/build.py
# по желанию, ~2 мин GPU: подтвердить фикс чипов на T4 и получить peak_vram_gb
KAGGLE_API_TOKEN=$(cat ~/.kaggle/access_token) ~/.local/bin/kaggle kernels push -p /workspace/koz3_shared/bot-9/kaggle/dfine_spor/smoke
# эта неделя (фолды 4,2,1, ≈3,7 ч, потолок 5 ч); не одновременно с другим 2×T4-ядром (риск 7)
KAGGLE_API_TOKEN=$(cat ~/.kaggle/access_token) ~/.local/bin/kaggle kernels push -p /workspace/koz3_shared/bot-9/kaggle/dfine_spor/full
# следующая неделя (фолды 0,3, ≈1,9 ч)
KAGGLE_API_TOKEN=$(cat ~/.kaggle/access_token) ~/.local/bin/kaggle kernels push -p /workspace/koz3_shared/bot-9/kaggle/dfine_spor/full_rest
```

## Риски
1. NaN в fp16 / DN-ветке → `DFINE_NUM_DN=0` (HF PR #45601 чинит aux-лоссы при DN=0; в transformers 5.21 не проверял).
2. Принудительная установка transformers==5.21.0 может конфликтовать с huggingface_hub/tokenizers образа Kaggle. На smoke v1 импорт прошёл (`koz3-dfine-spor-smoke.log` стр. 6). Ранний фейл на импорте, ассерт версии и «Шаг 2б» поймают проблему в первые минуты.
3. Окна только с фоном: при num_boxes=1 VFL-loss большой (~580). Смягчено долей объектных окон 60 % и clip 0.1.
4. Инференс полных растров: резерв ~20 мин на фолд может не хватить на больших участках. Тогда урезается обучение следующих фолдов; есть общий дедлайн и лог прогресса каждые 200 окон.
5. Городища крупнее окна 320 м не поместятся в рамку. Для них детектор слаб: ожидаемо, лидара нет.
6. Чипы JPEG в `/tmp/chips`: ≤400 объектных + фон на участок, на диске порядка единиц ГБ. Не замерено на полном наборе.
7. Двух GPU-ядер одновременно запускать нельзя: второе молча получит CPU и упадёт по `--require-gpu`.
8. Веса — только для замера: в сдачу после письма организаторов (п. «Веса и лицензия»).
9. С тренером бота 5 (по умолчанию) число итераций не пересчитывается на 50-й итерации, как у встроенного. Если на T4 не уложимся в бюджет фолда, адаптер остановит обучение по времени («[bot5] СТОП по времени»), и OneCycle не дойдёт до конца. Решение принято: по smoke v2 (0,92 с/ит) в full и full_rest зашито 6000 ит., чтобы OneCycle прошёл полностью (см. «Бюджет»).
10. Время чипов на полном наборе (5,4 ГБ tif, 132 участка) не замерено — оно сдвигает старт. Бюджет фолдов пересчитывается от текущего времени, так что запас просто уменьшается.

## full_rest готов к пушу (22:45 МСК 28.10, бот 9)
- Решение пользователя (через бота 1): все 5 фолдов на этой неделе. full_rest (фолды 0,3) пушится сразу после COMPLETE full, до koz3-cnx-li-smoke.
- Текущий full **не трогаем**: full_rest — отдельный кернел `dmitrijkv/koz3-dfine-spor-full-rest`, собран тем же `build.py` из того же кода. Отличия от full (`diff`): только `FOLDS_DEFAULT='0,3'` и `DEADLINE_H_DEFAULT=3.5` (прогноз ≈1,9 ч). sha256 c396346c9d48…, `py_compile` OK.
- Пуш (делает бот 1, только после COMPLETE full — одновременно два GPU-кернела нельзя):
  `KAGGLE_API_TOKEN=$(cat ~/.kaggle/access_token) ~/.local/bin/kaggle kernels push -p /workspace/koz3_shared/bot-9/kaggle/dfine_spor/full_rest`
- Если full упадёт и будет исправление — full_rest пересобирается тем же `build.py` вместе с ним; тогда эта отметка обновится.
- Квота: 3,7 + 1,9 ≈ 5,6 GPU-ч D-FINE на этой неделе.
- f4 (ConvNeXt) остаются как есть, ImageNet-вариант — только замер, не в сдачу до письма организаторов.
