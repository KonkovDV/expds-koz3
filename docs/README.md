# expds-koz3 — команда StrataTrace (КОЗ №3 «Экспедиция DS. Комплексирование»)

<!-- Оформлено по шаблону организаторов README.md (Прил. 5 утв. ТР). Обзор и текущее состояние — корневой README.md. -->

## 📋 Описание
Решение ищет археологические объекты семи классов (повреждённые и целые курганы, городища, архитектура, площадные объекты
геофизики, находки, фортификации) по лидару (облака LAS), спутниковым снимкам, историческому аэрофото, магнито- и
электроразведке и выдаёт один GeoJSON FeatureCollection в EPSG:3857 (только Polygon).

Архитектура — позднее слияние: каждый тип данных обрабатывает своя ветка, порог ветки снимается официальной метрикой
(Прил. 4) на участках, не входивших в обучение. Лидар: ЦМР строится из LAS (готовых хилшейдов Li на валидации нет), каналы
локального рельефа и уклона → U-Net ResNet-18 (карта центров для курганов, маски для городищ и фортификаций), запасной
детектор пиков + LightGBM. Снимки: U-Net ResNet-18 (ImageNet). Геофизика: классический детектор пиков, каждая карта даёт
свои объекты. Дубли сливаются только внутри одного типа данных. Тяжёлые шаги ЦМР — Rust (`expds_core`) с эквивалентом на NumPy.
Подробно — `docs/ARCHITECTURE.md`.

---

## ⚙️ Требования и установка

### Окружение
- **ОС:** Ubuntu 22.04 / Debian 12 или Windows 11
- **Docker:** образ `pantum/n7q:20261024` (сборка — `docker/Dockerfile`, база `python:3.12-slim-bookworm`; в образе только окружение)
- **Python:** 3.12
- **GPU:** NVIDIA с драйвером хоста; CUDA и cuDNN приходят в колёсах PyTorch (cu128). Состав и версии — `docs/THIRD_PARTY.md`

### Установка зависимостей
```bash
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.lock
pip install -r requirements-dev.txt      # для тестов и обучения
python scripts/build_rust.py dev          # для контейнера: python scripts/build_rust.py linux --docker
```

**Основные библиотеки:**
- `torch==2.11.0`, `torchvision==0.26.0` — нейросети
- `segmentation_models_pytorch==0.5.0`, `timm==1.0.30` — U-Net и энкодеры
- `laspy==2.7.0` — чтение LAS
- `rasterio==1.5.2`, `pyproj==3.8.0`, `shapely==2.2.0` — растры, СК, геометрия
- `numpy==2.5.4`, `scipy==1.18.1`, `lightgbm==4.7.0`
- _(полный список с точными версиями — `requirements.lock`)_

---

## 📊 Данные

### Датасет конкурса
1. **Скачивание:** с Платформы конкурса (выдаётся участникам; по п. 13.10 хранится только на инфраструктуре под контролем
   команды и не публикуется). Распаковать в любую папку `КОРЕНЬ/`.
2. **Предобработка:** эталон в общий GeoJSON и кэш тайлов по типу данных:
   ```bash
   python -m expds.tiles --root КОРЕНЬ --gt work/gt_train.geojson --markup li --out work/cache/li --res 0.5 --tile 256
   ```
   Для лидара каналы строятся из LAS той же функцией, что на инференсе.

### Дополнительные открытые датасеты
Сторонние обучающие данные не используются. Предобученные веса энкодеров:

| Источник | Описание | Лицензия | Скрипт загрузки |
|----------|----------|----------|-----------------|
| [ImageNet](https://www.image-net.org/) через `timm` / `torchvision` | Инициализация энкодера ResNet-18 | код — BSD-3 (torchvision) / Apache-2.0 (timm); набор ImageNet — условия для некоммерческих исследований; веса ImageNet допущены организаторами 29.10 (сводка `docs/STATUS_2026-10-29.md`; нужна письменная фиксация), см. `docs/THIRD_PARTY.md` | загружаются библиотекой при обучении; запасной путь — `--no-pretrained` |

> ⚠️ Сторонних обучающих данных нет (п. 13.5). Предобученные веса — только из допущенного организаторами списка; данные КОЗ №2 не используются. Без предобучения: `python -m expds.train --no-pretrained`.

---

## 📁 Структура проекта

```
.
├── metadata.json                 # образ Docker Hub и команда запуска (entry_point)
├── requirements.txt / requirements.lock
├── docker/Dockerfile             # окружение платформы
├── docs/
│   ├── README.md                 # этот файл
│   ├── ai_declaration.md         # декларация ИИ-инструментов (журнал — AI_DECLARATION.md)
│   ├── RULES.md                  # правила утв. ТР, касающиеся кода
│   └── COMPLIANCE_2026-10-29.md  # сверка кода с утв. ТР
├── configs/default.toml          # пороги, веса, флаги сдачи
├── models/                       # веса из конфига
├── expds/                        # пайплайн: каталог, СК, лидар, геофизика, сети, пороги, слияние, запись
├── rust/expds_core/              # LAS, ЦМР, отмывки, пики
├── inference/
│   ├── main.py                   # единственная точка входа
│   └── lib/                      # собранный expds_core
├── scripts/                      # сборка Rust, архив, проверка архива, честное табло
└── tests/                        # синтетические тесты (pytest)
```

---

## 🔧 Описание основных функций

### `expds/pipeline.py`
```python
run(input_dir, output_path, config_path=None, ...) -> ...
```
Полный прогон: каталог участков, ветки по типам данных, пороги, слияние, запись GeoJSON. Следит за временем (`expds/watchdog.py`, 30 мин).

```python
lidar_feats(reg, cfg, guard, seg_cache, root, ...) -> list[dict]
```
LAS → ЦМР → каналы → сети/запасной детектор; без LAS участок пропускается без ошибки.

```python
geophys_feats(reg, cfg, guard) -> list[dict]
```
Детектор геофизики; каждая карта (срез/профиль) даёт отдельные объекты.

### `expds/post.py`
```python
merge_within_markup(feats: list[dict]) -> list[dict]
```
Сливает дубли внутри ключа (участок, тип данных, класс); между типами данных не сливает, EtGr не сливается вовсе.

### `expds/writer.py`
```python
write_geojson(feats: list[dict], path) -> Path
validate(fc) -> dict
```
Запись в EPSG:3857 с лимитами Прил. 3 (≤ 3000 вершин, ≤ 20 000 объектов, ≤ 97 МБ, валидность после округления) и подсчёт замечаний валидатора.

### `expds/train.py`, `expds/metric.py`
```bash
python -m expds.train --cache ... --out models/li.pt ...
python -m expds.metric sweep --pred ... --gt ... --out thresholds.json
```
Обучение U-Net и подбор порогов официальной метрикой (`materials/compute_metrics.py`).

---

## 🚀 Инструкции по запуску

### Вариант 1: Локальный запуск

#### Обучение модели
```bash
python -m expds.tiles --root КОРЕНЬ --gt work/gt_train.geojson --markup li --out work/cache/li --res 0.5 --tile 256
python -m expds.train --cache work/cache/li --out models/li.pt --epochs 30 --batch 16 --val-families СЕМЬЯ --workers 0
```

#### Инференс
```bash
python inference/main.py ПАПКА_ВЫБОРКИ work/result.geojson
python -m expds.writer check work/result.geojson
```

### Вариант 2: Docker

#### Сборка образа
```bash
docker build -t solution:latest -f docker/Dockerfile .
```

#### Запуск инференса
```bash
python scripts/build_submission.py          # dist/submission.zip ≤ 2 ГБ: код, конфиг, веса, metadata.json
python scripts/check_submission.py dist/submission.zip
docker run --rm --gpus all --network none -v $(pwd)/dist/unpacked:/app -v ВЫБОРКА:/data:ro -w /app \
  solution:latest python inference/main.py /data /app/result.geojson
```

---

## 🔍 Воспроизведение решения

```bash
# 1. Установка зависимостей (см. выше) и сборка Rust
python scripts/build_rust.py dev
# 2. Подготовка данных: эталон и тайлы по типам данных
python -m expds.tiles --root КОРЕНЬ --gt work/gt_train.geojson --markup li --out work/cache/li --res 0.5 --tile 256
# 3. Обучение сетей (по каждому типу данных) и подбор порогов на отложенных семьях участков
python -m expds.train --cache work/cache/li --out models/li.pt --epochs 30 --batch 16 --val-families СЕМЬЯ --workers 0
EXPDS_METRIC=materials/compute_metrics.py python -m expds.metric sweep --pred work/oof.geojson --gt work/gt_train.geojson --markup li --out work/thresholds.json
python scripts/scoreboard.py --honest 10
# 4. Инференс и сборка архива
python inference/main.py ПАПКА_ВЫБОРКИ work/result.geojson
python scripts/build_submission.py && python scripts/check_submission.py dist/submission.zip
```

---

## 📧 Контакты
Команда: StrataTrace  
Email: _заполнить перед подачей_
