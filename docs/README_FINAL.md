# expds-koz3 — команда StrataTrace (КОЗ №3 «Экспедиция DS. Комплексирование»)

<!-- Финальная версия по шаблону организаторов README.md (Прил. 5 утв. ТР, п. 8.5). Черновик 30.10.2026:
     перед подачей сверить флаги с configs/default.toml сдаваемой версии и таблицу весов с docs/WEIGHTS_MANIFEST_v4.md. -->

## 📋 Описание
Поиск археологических объектов по каждому типу данных участка с выдачей полигонов GeoJSON EPSG:3857 (Прил. 3 ТР).
Архитектура — **позднее слияние**: у каждой модальности свой детектор, в пределах одного типа данных объект — один
полигон (дедупликация), между типами данных полигоны **не склеиваются** (метрика считает каждый тип отдельно).

| Модальность | Детектор | Классы | Статус во флаге |
|---|---|---|---|
| Лидар (La → DEM 0,1/0,5 м, Li-карты) | U-Net ResNet-18 (`li_r18_hard2`, `li_gor_mask_r18`, `li_fort_mask_r18`, `li_dev_r18`) + мозаика DEM; ConvNeXt U-Net c3-sc f0/f1 (`[li_cnx]`, каналы рельефа + RVT SVF/открытость) | курганы, городища, фортификация, архитектура | r18 — вкл.; `li_cnx` — выкл. до гейта |
| Курганы | центры курганов (фокальная карта) + LightGBM-оценщик кандидатов (`kurgan_lgbm`), сиды курганов W3 | курганы целые/повреждённые | вкл.; новые сиды — по гейту |
| Спутник SpOr | U-Net r18 (`spor_focal_r18`, `spor_focal_wide_r18`, `spor_gor_r18`, `spor_arch_r18`, `sp_crop`); D-FINE-L (`[sp_dfine]`) | все 5 классов рельефа | r18 — вкл.; D-FINE — выкл. до гейта |
| Аэрофото Ae / ортофото Or | `ae_fort_mask_r18`, `surf_arch_r18`; D-FINE-L Ae/Or (`[dfine_aeor]`, 3 фолда Or) | фортификация, архитектура, городища | D-FINE — выкл. до гейта |
| Геофизика MagGr / EtGr / TerFl | классические детекторы `expds/geophys.py` (CPU); порт бейзлайнов организаторов: EtGr — контуры по картам и профилям + 16 признаков + HGB + NMS (GEO-ORG3), MagGr — многоуровневые полигоны (GEO-ORG1) | finds_points, object_poly, structure_poly | GEO-ORG1/3 — в работе (бот 5), включение по гейту |
| Городища (все слои) | шлюз DINOv2-S + своя голова (`[gor_gate]`, `[gor_dense]`) | городища | выкл. до гейта |
| Переранжировщик | LightGBM `rerank_arch_lgbm` по признакам кандидата (+ эмбеддинги DINOv2 — в работе) | архитектура | вкл. |
| Постобработка | пороги по классу и типу данных (честный подбор на honest-10), дедупликация, `[post] micro_circles` с якорем во внутренней точке исходного полигона (rp-anchor, +0,0044 на honest-10) | все | `micro_circles rp-anchor` — выкл. до переподбора порогов |

## ⚙️ Требования и установка
- **Платформа:** Docker-образ `pantum/n7q:20261024` (публичный, Docker Hub; только окружение, без кода и весов), `metadata.json` в корне архива: `"entry_point": "python inference/main.py"`.
- **База образа:** `python:3.12-slim-bookworm`, PyTorch 2.11.0 + torchvision 0.26.0 (CUDA 12.8), RTX 4090.
- **Зависимости:** точные версии — `requirements.lock`; нижние границы для разработки — `requirements.txt`. Основные: torch, segmentation-models-pytorch 0.5.0, timm 1.0.30, lightgbm, rasterio, shapely, pyproj, laspy/lazrs, scipy. Ядра каналов лидара — Rust (`rust/expds_core` → `inference/lib/expds_core*.so`).
```bash
docker build -t solution:latest -f docker/Dockerfile .
pip install -r requirements.lock   # локально, без Docker
```

## 📊 Данные
- **Датасет конкурса:** обучающая выборка организаторов (Я.Диск / S3 `train-expds-3`, ключи — у организаторов, в репозитории не хранятся). Апдейты датасета до 20.10.2026 учтены. По п. 13.10 ТР данные не публикуются и не выносятся за пределы инфраструктуры команды (локальный бокс, приватные датасеты Kaggle).
- **Валидация:** honest-10 — 10 фолдов по участкам/семьям участков; пороги каждой семьи подбираются на остальных; метрика — официальный `compute_metrics.py`.
- **Внешние открытые датасеты:** не используются. Предобучение весов: ImageNet (ResNet-18, ConvNeXt), COCO (D-FINE-L), LVD-142M (DINOv2). Лицензии — `docs/THIRD_PARTY.md`, `docs/ai_declaration.md` §2а.

## 📁 Структура проекта
```
.
├── metadata.json            # образ и точка входа
├── docker/Dockerfile        # окружение (публичный образ)
├── requirements.txt / requirements.lock
├── inference/main.py        # единственная точка запуска
├── inference/cnx/           # ConvNeXt-лидар (WEIGHTS_MANIFEST.json, вендор rvt)
├── expds/                   # пайплайн: catalog, lidar, geophys, infer_nn, post, writer, …
├── configs/default.toml     # флаги и пороги сдачи
├── models/                  # веса (большие — вне git, sha256 ниже)
├── rust/expds_core/         # Rust-ядра каналов DEM
├── scripts/                 # обучение, OOF, сборка архива (build_submission.py)
├── tests/
└── docs/                    # README_FINAL.md, ai_declaration.md, THIRD_PARTY.md, …
```
Архив для платформы собирает `scripts/build_submission.py` (только код инференса, конфиги и включённые флагами веса; ≤ 2 ГБ; документы в архив платформы не входят, для экспертизы передаётся весь репозиторий).

## 🔧 Основные функции
- `inference/main.py` — `python inference/main.py <папка_выборки> <result.geojson>` (или `EXPDS_INPUT`/`EXPDS_OUTPUT`, иначе `/data/input` → `/data/output/result.geojson`).
- `expds/catalog.py` — разбор папок участков (`La/Li/Ae/SpOr/Or/MagGr/EtGr карты и профили/TerFl`, UTM.json); неизвестные модальности (георадар) пропускаются.
- `expds/pipeline.py` — запуск детекторов по флагам `configs/default.toml`, сторож времени.
- `expds/writer.py` — последний рубеж Прил. 3: только Polygon, ≤ 3000 вершин, ≤ 20 000 объектов, ≤ 100 МБ, EPSG:3857.

## 🚀 Запуск
```bash
# обучение (пример: U-Net лидара; остальные — scripts/* и Kaggle-ядра в notes/)
python -m expds.train --config configs/default.toml
# инференс локально
python inference/main.py /path/to/val result.geojson
# Docker
docker run --rm --gpus all -v $PWD/val:/data/input -v $PWD/out:/data/output pantum/n7q:20261024 python inference/main.py
```

## ⏱ Время в контейнере (лимит 30 мин)
`[time] budget_s = 1500` (25 мин), сторож пишет найденное за 3 мин до 30 мин (1620 с). Последние посылки на публичной выборке: 3 мин 11 с – 3 мин 22 с (`docs/LB_LOG.md`, 24–27.10). Новые тяжёлые модули (D-FINE, ConvNeXt, DINOv2) включаются только после замера времени на 4090.

## 🧮 Веса (sha256; источник — `docs/WEIGHTS_MANIFEST_v4.md`, `inference/cnx/WEIGHTS_MANIFEST.json`)
| Файл | sha256 |
|---|---|
| models/li_r18_hard2.pt | 0311e877e148e0ec3acb6f3b13a7fb12affe5e0b58a60d3524a2df956b3793b1 |
| models/li_r18_hard.pt | 100bcc21afe8fc09be5362e19e2621ec815db28a38cad82a6bd9a5dffa7cc7d9 |
| models/li_r18_all.pt | 2ed166cd9b37c3514438b885ec196cd6c54aeba9331bb5a4e81c55faa71ef531 |
| models/li_dev_r18.pt | 2b655c809cb89c9394fb36ae9fec1d710b1d17d35a671834bdd80404b28265ab |
| models/li_fort_mask_r18.pt | eb11cff8be474501d26fb6755268150abeaed42289512a6dd507b0571d4934e9 |
| models/li_gor_mask_r18.pt | 39702cd8a5d92d0c0c06132690febc265519ed7a35b3c6ff02ed1fd7c1fdd8ef |
| models/ae_fort_mask_r18.pt | 36d64e1a8a14d850dbf1494a11629e2a4ff7cea218b10dd462c5b827cf58bba7 |
| models/spor_gor_r18.pt | 00d5c08539d1bced03d373c018d9e31432d987c7d848196a4afac6b0d44f1cbe |
| models/spor_focal_r18.pt | 88bbd1a24bf4bd4a2dae8c7855a95767e117124955273c9df1b2255c96fdd88b |
| models/spor_arch_r18.pt | 9478716f2b41567a25733919a92d55f2f199c8ca1d09c337ac2571705bb30cac |
| models/spor_focal_wide_r18.pt | aa09855cb95babc8bfaf8018e6d32763ff0fd8356db5c4fb41fa31333a4f742c |
| models/surf_arch_r18.pt | 8266add934346e5c675cda0c4aeb84ee39be3652e9308136a789673d768011a7 |
| models/sp_crop.pt | 4a41b0caada94d387721eb13526071469d5c70504b9dfb6743e7923a1f39cbeb |
| models/kurgan_lgbm.txt | f249a47e3305f21ac6bb45f3f35169019c52b93b53574d572b12ea28542e796c |
| models/rerank_arch_lgbm.txt | 0bc2e924b7a7beb79706eae275d516a2668406294e99b6caf44e454030b6d97b |
| models/rerank_arch_lgbm.json | d76c7edc54fc1c377a652122dcc8c7022e62244d48a6c0bbe85ecc19d2215ab0 |
| models/cnx/weights_cnx_c3-sc_f0.pt | e84af54a9d110842fe22223e26beed03bdc0c01a383281a0ead360bd69000157 |
| models/cnx/weights_cnx_c3-sc_f1.pt | fc6a03b637e11418d0f6bef2a1ae333f5d5aeb8c8f80dd6c1eb33381ba7f02e4 |
| models/dfine_f0/model.safetensors (SpOr) | 81ef82280c43dd5e48255587f449683603ece375c976981db4f9a243e8509ed8 |
| models/dfine_or/dfine_f0/model.safetensors | 82c8a47a2ad0d0b12fd5467b38592d31910114ad4029242d933f2e807c517d9c |
| models/dfine_or/dfine_f1/model.safetensors | 7fcb58645366819e83f125b0078a5fdc0b4b896f1c2a1b0b3c995820586b6744 |
| models/dfine_or/dfine_f2/model.safetensors | 29dcd3c633c31aa153abe420bde59766de25fa662b4b2b3065f7a78dacd0ed2e |
| models/gor_gate/gor_v3_head.npz | 5bd88f69b3ffbfdc5312214b4d6012c92be12c66539b7bf8ad35564c76a75e7d |
| models/gor_dense/gor_dense_head.npz | feb1591bb1e9ada63b32829178eae44a7e19fdfd7e99d7d222e3b1389c5b2cb7 |
| models/dfine_ae/* | заполнить после обучения (W1, `koz3-w1-dfine-ae-final`) |
| models/pixgbm/all/{poly,finds}.txt | **не включать**: модуль `geo_pixgbm.py` написан по мотивам чужого решения, заменяется clean-room/GEO-ORG1 |

## 🔍 Воспроизведение
```bash
pip install -r requirements.lock
cargo build --release --manifest-path rust/expds_core/Cargo.toml   # ядра DEM
python scripts/oof_full.py            # OOF и честные пороги (honest-10)
python scripts/build_submission.py    # архив для платформы
```

## 📧 Контакты
Команда: StrataTrace  
Email: заполнить
