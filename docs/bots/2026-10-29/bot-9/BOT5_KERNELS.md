# BOT5_KERNELS — кернелы Kaggle для трёх заданий bot-5 (bot-9, 29.10.2026)

Пушит bot 1. Я ничего не пушил и не запускал на Kaggle, в GitHub тоже ничего не пушил. Репо bot-5 только читал (`git show <commit>:<path>`).
Все кернелы: пользователь `dmitrijkv`, private, docker-пин как у `kaggle/dfine_spor/final/kernel-metadata.json`.
Ключи S3 берутся только из Kaggle Secrets `KOZ3_S3_*` → env → датасета `dmitrijkv/koz3-s3-cred`, как в остальных кернелах bot-9.
Ключей нет ни в коде, ни здесь. «Сканирование» (ackrome/satelite-scan) нигде не используется: `grep -rnE "satelite-scan|ackrome"` находит только строки с запретом и проверку в build.py.

Команда пуша (для каждого каталога):
`KAGGLE_API_TOKEN=$(cat ~/.kaggle/access_token) ~/.local/bin/kaggle kernels push -p <каталог>`

| № | задание | статус |
|---|---|---|
| 1 | плотный локализатор городищ (FOR_BOT9_GORLOC, @6984c67) | **ГОТОВО** (1 GPU-кернел + 3 CPU-кернела, smoke по желанию) |
| 2 | RF-DETR-M O365→COCO (FOR_BOT9_RFDETR, @c43cb08) | **БЛОКЕР**: нет датасета `koz3-spor-det05` (см. ниже) |
| 3 | DINOv2-reg B/L (FOR_BOT9_DINOV2REG, @a0ac2ef) | **ЧАСТЬ A ГОТОВА**; часть B — блокер (нет STACK/GT на Kaggle) |

---
## 1. Плотный локализатор городищ: ГОТОВО

Код bot-5 не менялся. `dense_feats.py`, `dense_probe.py`, `expds/{__init__,constants,naming,metric}.py` и `kaggle/gor_dense_inputs.json` встроены в скрипт (gzip+base64).
При распаковке sha256 сверяется. Хэши лежат в `kaggle/bot5/gorloc/MANIFEST.json`; inputs json совпадает с `bot-5/analysis/gor_dense/gor_dense_inputs.json`.
Сборка: `kaggle/bot5/gorloc/build.py`, шаблоны в `gorloc/tpl/`.

**Чем отличается от ноутбука bot-5.** Ноутбук ждёт датасеты `expds-koz3-bot5` и `train-expds-3`, а под нашим пользователем их нет. Поэтому:
- данные берутся из S3. Это ровно 175 SpOr-tif и 132 UTM.json, 5,40 ГБ (`s3/listing.txt`; такой же фильтр по листингу даёт 175/132). В режиме full это проверяется assert'ом;
- работа разбита на GPU-кернел признаков и CPU-кернелы пробы через `kernel_sources`. Проба `dense_probe.py` идёт на CPU, и в одну 12-часовую GPU-сессию _L12 не помещается (оценка ниже).

| каталог `kaggle/bot5/gorloc/kernels/…` | что делает | машина | sha256 .py | sha256 metadata |
|---|---|---|---|---|
| `koz3-b5-gorloc-smoke` (необязателен) | smoke bot-5: 185_Изборск, GSDS=1,2, F.shape[1]==2048, всё конечное | T4 | `03359e8958cfb539d1929465a21b28a20632071f2be3192213f4a70f00e9598a` | `6ec9d92993f8b96aba20adb56032d6b42fcb3744144d3dc2aca75eb4190803ae` |
| `koz3-b5-gorloc-feats` | тот же smoke как гейт (при ошибке СТОП), затем признаки `_L24` (2/4 м/пкс) и `_L12` (1/2 м/пкс) → `/kaggle/working/gd/` | T4, internet | `e273f42cc3aee105910be7acfb5966115b70ed1636a99297891e36cb04369f30` | `ec69ae45241ae9dce0dbbf6eac0d4029b624bac3862ef1f117cdaa4929309315` |
| `koz3-b5-gorloc-probe-l24` | `dense_probe` _L24: fam10, seed0–4, LOFO → `peaks_L24.json`, `gor_dense_L24_all.tgz` | CPU, источник: feats | `0b6859c170d8f3d206a02729acfd5b13a72f47bc0d3095ea90d5696f3286d559` | `bc9844751de0cc172512856e5f68875266a4b5a972f9c08683d488a74c8f2c3c` |
| `koz3-b5-gorloc-probe-l12a` | _L12: fam10 + seed0–4 → `gor_dense_L12_fam10seeds.tgz` | CPU | `4418231ca096556591f6da4f359da0bcb5b5f4137587fb3b1b8c24027c155b42` | `e0ecdd30dab5e1ff83ed57614d22247602838f4bb14b0fa60568778d5ec00e56` |
| `koz3-b5-gorloc-probe-l12b` | _L12: LOFO → `gor_dense_L12_lofo.tgz` | CPU | `9463229b8dcc810d3bc0af758599e77d25c3ee722cb252b3c024366ed81438d3` | `c82411aec94fd80c9c48ea55b86e9079d23d124236f9a892682e8d354a0e5e1c` |

**Порядок.**
1. (по желанию) smoke.
2. `koz3-b5-gorloc-feats`.
3. Когда feats успешно завершится, все три probe-кернела. Это CPU-кернелы: GPU-очередь они не занимают, и их можно запускать параллельно со следующей GPU-задачей.

**Возврат bot-5.**
- `tar xzf` архивов в `analysis/gor_dense`.
- _L12 склеить: `python kaggle/bot5/gorloc/merge_peaks.py peaks_L12.json <peaks_L12 из l12a> <peaks_L12 из l12b>` (sha256 `f535c0f1447bc186318da95932c739b7b42aa5c21548c43e976a7802763f546f`).
- Затем `DTAG=_L12 python scripts/gor_dense/gor_dense_eval.py` (и `_L24`), как в FOR_BOT9_GORLOC.

**Проверки.**
- Честные фолды: раскрои fam10, LOFO(103), seed0–4 покрывают 210 участков и 103 семьи. Ни одна семья не разрезана (проверял `site_family` в bot-9/venv). Непустых отложенных фолдов: fam10 10, LOFO 58, каждый seed 10. Проба учится только внутри фолда, это код bot-5 без изменений.
- Формат вывода: на одном раскрое склеенный `peaks_L24.json` побайтно равен результату `dense_probe.py`. На двух раскроях (отдельные процессы против одного) `layouts` и `stats` идентичны. `hn_log` и `cp_s_chosen` — это счётчики-суммы. Они немного больше (288 фитов против 270), потому что у отдельных процессов нет общего `_cache`. На метрики это не влияет (`gorloc/dry_logs/probe_dry2.log`, `ref.log`).
- Сухой прогон на CPU: крошечная случайная Dinov2WithRegisters (hidden 64), 13 участков из 12 семей (`dry_logs/regs.txt`). Smoke, _L24 и _L12 дали rc=0, F конечный, dim совпал (`dry_logs/feats_dry.log`). Проба fam10 rc=0, пиковый RSS 146 МБ (`dry_logs/probe_dry.log`). Настоящие веса на бокс не скачивались. Сухой прогон feats был до последней правки (THREADS берётся из env B9_THREADS). Правка однострочная, py_compile проходит.
- Веса: `facebook/dinov2-with-registers-large@e4c89a4e…`, apache-2.0 (HF API). DINOv2-reg входит в список разрешённых.
- Пины: transformers 5.21.0, rasterio 1.5.2, pyproj 3.8.0, shapely 2.2.0 (`requirements_gor.txt` bot-5), boto3 1.40.61. У всех есть колёса cp313, ставятся с `--only-binary`. Internet нужен для pip и HF.

**Часы.**
- Оценка bot-5: 1,5–3 ч одной T4-сессии (FOR_BOT9_GORLOC.md).
- Моя оценка GPU: feats ≤ 1 ч T4. Это оценка, не замер. В неё входят скачивание 5,40 ГБ, чтение tif, на CPU-прототипе 580 тайлов _L24 (`bot-5/analysis/gor_dense/dense_feats.log`), а у _L12 тайлов примерно в 2,5 раза больше (сухой прогон: 35 → 89 тайлов). Сам ViT-L fp16 займёт минуты.
- Моя оценка CPU для пробы, оценка, не замер: ~110 с на фолд на боксе bot-5 (`bot-5/analysis/gor_dense/STATUS.md:37`). Токенов в _L12 в 4,05 раза больше, чем в _L24 (сухой прогон: 43 741 против 10 792). Отсюда:
  - _L24 ≈ 118 фолдов × 110 с ≈ 3,6 ч;
  - _L12-a ≈ 60 × ~440 с ≈ 7,3 ч;
  - _L12-b (LOFO) ≈ 58 × ~440 с ≈ 7,1 ч.
- Вывод: если пробу гонять внутри GPU-сессии, 1,5–3 ч — это заниженная цифра.

**Ожидаемый прирост (оценка bot-5, не наша).** Ориентир по приору рамки 0,279→0,297 взят из аудита конкурента, это не наш замер (FOR_BOT9_GORLOC.md).

**Риски.**
1. LOFO _L12 — это один раскрой, а `dense_probe` пишет результат только после целого раскроя. Если раскрой не уложится в дедлайн 11,5 ч, LOFO _L12 потеряется целиком. fam10/seed-части сохраняются по одной.
2. Kaggle-CPU (4 vCPU) может оказаться медленнее или быстрее бокса bot-5, замера нет.
3. kernel_sources требуют успешной версии feats. При частичном выходе feats завершается с exit 1, и probe-кернелы надо будет пушить уже после перезапуска.
4. Признаки _L12 в fp16 занимают ~3,7 ГБ (≈920 тыс. токенов × 2048). Укладывается в 20 ГБ вывода и в ОЗУ CPU-кернела.

**Вопросы bot-5.**
- (а) Согласен ли ты на разбиение GPU-признаки → CPU-проба и на склейку `merge_peaks.py`?
- (б) Можно ли добавить в `dense_probe` выбор подмножества фолдов (или сохранять после каждого фолда), чтобы LOFO _L12 можно было резать? Есть и подозрение на горячую точку: сравнение строк `self.REG[dtr] == r` в цикле.
- (в) Нужен ли отдельный smoke-кернел, или достаточно встроенного гейта в feats?

---
## 2. RF-DETR: БЛОКЕР (данные)

Код: `c43cb08` (ветка bot5-rfdetr-coco). Файлы: `scripts/rfdetr_train.py`, `kaggle/rfdetr_folds.sh`, `README_KAGGLE_RFDETR.md`.

**Блокер.**
- Ноутбук требует датасеты `koz3-spor-det05` (f0..f4, grid_f0..4 из `coco_from_cache.py --layout rfdetr`) и `expds-koz3-bot5`.
- Под `dmitrijkv` их нет. Кэша `spor_det05` нет нигде на боксе (`find /workspace -maxdepth 5 -name 'spor_det05*' -o -name grid_f0` ничего не нашёл).
- В FOR_BOT9 написано «те же, что для D-FINE», но наш кернел D-FINE (`kaggle/dfine_spor/final`) режет окна сам из S3 и `koz3-spor-det05` никогда не строил.
- Чтобы построить `det/`, нужны `python -m expds.tiles … --gt work/gt_train_v2.geojson --markup sp_or` (×2, окна и сетка) и `coco_from_cache.py` ×10 (README_KAGGLE.md, шаг 0).
- `work/gt_train_v2.geojson` в bot-5 нет. Нашёлся `/workspace/rerank/gt_train_v2.geojson` (39 МБ, sha256 `d4bc1a3b01350c9b…`), но это ли та версия, не знаю.

**Что проверено.**
- Фолды честные: `split_rows` → `family_folds(regions, site_family, 5)` (`coco_from_cache.py:126-146`).
- Веса A `rf-detr-medium.pth` и B `rf-detr-base-o365.pth` — Apache-2.0 по реестру rfdetr 1.11.2. COCO и Objects365 разрешены. md5 указаны в README bot-5. Чекпойнты качаются с GCS, поэтому нужен internet.
- 2 полосы GPU соответствуют T4×2 на Kaggle (`machine_shape NvidiaTeslaT4`, как у D-FINE).

**T4-часы.** У bot-5 только план: 9 ч на сессию (`DETECTORS_SOTA.md:302`), замера нет. Точную цифру даёт формула bot-5 после smoke: `s/ит(bs4) × 4 × ITERS × 3 + сетка`.

**Ожидаемый прирост (оценка bot-5).** В FOR_BOT9_RFDETR не указан.

**Вопросы bot-5.**
- (а) Кто и где строит `det/`? Варианты: ты на своём боксе, после чего bot 1 заливает приватный датасет. Либо я делаю CPU-кернел подготовки из S3, но тогда GT всё равно нужно доставить на Kaggle датасетом (39 МБ).
- (б) Какой файл GT правильный? Совпадает ли он с `/workspace/rerank/gt_train_v2.geojson`?
- (в) Работает ли `expds.tiles` (`scan_root`), если скачан только SpOr? Без ответа я не гадаю и кернел не собираю.

---
## 3. DINOv2-reg: ЧАСТЬ A ГОТОВА, часть B — блокер

Каталог: `kaggle/bot5/dinov2reg/kernels/koz3-b5-dinov2reg-gorpatch` (T4, internet, датасет koz3-s3-cred).
- sha256 .py `7c6b8642eb25529b1c73e709921e7b6cd90b11c8dfd6a75ae2df03b1d2e8be0f`
- sha256 metadata `57017b419fc815280ea2066e6628b863c36a8143e71e4d47d130930aee8466f4`
- Сборка: `dinov2reg/build.py`. Встроены `scripts/gor_patch/extract_feats.py` и `scripts/det_train_hf.py` @a0ac2ef (sha в `dinov2reg/MANIFEST.json`).

**Как работает.**
1. Веса B `a1d738cc…` и L `e4c89a4e…`, обе apache-2.0 (HF API, проверено).
2. Из S3 скачиваются все SpOr (assert 175/132).
3. Smoke bot-5: LIMIT_FILES=2 для B и L, assert `F.shape[1]==4×hidden`, всё конечное, meta=N. При ошибке СТОП.
4. Полные прогоны B, затем L.
5. Выход: `/kaggle/working/gor_patch_reg.tgz` (`feats_reg{B,L}.npz`, `meta_reg{B,L}.json`) и `run_summary.json`. Если число виньеток ≠ 3982 (как у DINOv2-S, `bot-5/analysis/gor_patch/extract_feats.log`), в лог пишется предупреждение.

**Возврат bot-5.** `tar xzf gor_patch_reg.tgz -C analysis/gor_patch` → `FEATS_TAG=_regB` / `_regL` `python scripts/gor_patch/gor_patch_eval.py`.

**Проверки.**
- Признаки замороженного ViT без учителя, утечки на этапе извлечения нет. Фолды fam10/LOFO/seed считает eval bot-5.
- Сухой прогон: крошечная модель, 3 участка через симлинки. smoke_B/L, B и L дали rc=0, архив с 4 файлами (`dinov2reg/dry_logs/`).

**T4-часы.**
- Оценка bot-5: 15–30 мин на модель, часть A + B 1–2,5 ч.
- Моя оценка части A: ≤ 1 ч T4, не замер. Это ≈3982×2 прохода ViT на 224 px на модель, скачивание 5,40 ГБ, а чтение tif на CPU у bot-5 заняло 644 с за все 175 файлов вместе с DINOv2-S.

**Ожидаемый прирост.** В FOR_BOT9_DINOV2REG не указан.

**Часть B (кроп-верификатор Or): БЛОКЕР.** Ноутбук ждёт `/kaggle/input/expds3-stack/{stack_base.geojson, gt_train_v2.geojson}`, а таких входов на Kaggle нет. Кроме того, `crop_verify.py --markup or` нужны Or-растры, это отдельный объём из S3.
- Вопрос bot-5: откуда брать STACK (`scoreboard.py --dump-stack`) и GT? Подойдёт ли, если залить их приватным датасетом (заливает bot 1)?

**Риск.** Задания 1 и 3A скачивают одни и те же 5,40 ГБ, по ~10–15 мин на сессию. Объединять их я не стал, чтобы не смешивать приоритеты.

## Обновление 29.10 ~09:15 (bot-5 ANSWERS_BOT9_REVIEW)

### gorloc probes пересобраны @4a39427
Feats/smoke не трогали (sha feats e273f42c… как в очереди бота 1).
Вшиты dense_probe abbe4ea8… и peaks_ckpt 36b864cc…. LOFO → два ядра.

| ядро | .py sha256 |
|---|---|
| koz3-b5-gorloc-probe-l24 | ea50d3f40f9051fe7645a96d977326e3a942a5b5ad6230e305f838e5bbbfefd7 |
| koz3-b5-gorloc-probe-l12a | 9762baeb67cb694cdf55fa9d045f3290054a6fab0527faab1ffafe3d1c70235d |
| koz3-b5-gorloc-probe-l12b0 (FOLDS=0-51) | 1313a6dd97f439f78bb3c9da43cd1ba7fad16fc4f60749d784209739fa68e3d0 |
| koz3-b5-gorloc-probe-l12b1 (FOLDS=52-102) | 401f079df019b30090c44ed0a7de410662051f3f7892da41f998cc5c2ca7cfdd |

Пуш (CPU, после COMPLETE feats). Старый probe-l12b — не пушить (…OLD_do_not_push).
Слияние LOFO: python peaks_ckpt.py CK0,CK1 OUT.json

### RF-DETR
GT confirmed: /workspace/rerank/gt_train_v2.geojson sha256 d4bc1a3b01350c9b3785155dc4d7c13403fdfe7a098664bcb259f96bc168ff86.
Private dataset для бота 1: залить этот файл как gz (~11.5 МБ), id предлагаю dmitrijkv/koz3-gt-train-v2.
Ядро со сборкой det05 внутри — в работе (не READY).

### dinov2reg-B
WAITING npz / новую версию от бота 5 (вариант B).

## 29.10 ~09:40 — RF-DETR и dinov2reg-B (Or-кропы)
- ВАЖНО: `kaggle kernels push` заливает только code_file → исходники встроены в .py (`kaggle/bot5/embed_src.py`).
- RF-DETR `kaggle/bot5/rfdetr/kernels/koz3-b5-rfdetr/`: py b865e77e…43bb, meta 92033327…ffb2, src c43cb08 tar ba892509…1770 (встроен). Датасеты koz3-gt-train-v2 + koz3-s3-cred. Smoke: B9_MODE=smoke.
- dinov2reg-B `kaggle/bot5/dinov2reg/kernels/koz3-b5-dinov2reg-orcrops/`: py 2b43a0e6…10ae, meta 91c3c6af…b25a; src f7cf5d2 (expds + crop_verify ca214839… + rerank_oof) tar 58efe695…6586 встроен. Вход: датасет dmitrijkv/koz3-b5-or-arch-crops (or_arch_crops.npz sha 6663649d…d44c, assert). Энкодеры resnet18 / DINOv2-S@ed25f3a3 / regB@a1d738cc / regL@e4c89a4e → gp/p_or_<tag>.{npz,json} → verify_or_reg.tgz. CPU dry-run (DINOv2-S): распаковка, sha, загрузка весов и эмбеддинг кропов идут (128/626 за ~4 мин на 2 потоках), остановлен — до конца не доводил.
