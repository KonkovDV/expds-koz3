# Сторонний код, модели и данные

Требования: ТР п. 12.3–12.5 (соблюдать открытые лицензии, не мешать передаче прав Оператору), п. 13.4 (нельзя код и библиотеки с ограничением некоммерческого использования), п. 13.5 (сторонние данные — со ссылками и подтверждением лицензии).

Код и веса других команд в это дерево не входят. 26.10 их публичные репозитории читались отдельно: общая идея карты центров у нас уже была, их метрика пересечения полигонов в EPSG:3857 не используется, YOLO не берётся. Из литературы в код добавлены только необязательные каналы `rough` и `curv`.

## Rust (rust/expds_core, Cargo.lock — 66 крейтов)

| Крейт | Лицензия | Назначение |
|---|---|---|
| pyo3 0.30 | MIT OR Apache-2.0 | привязка к Python |
| numpy (rust-numpy) 0.30 | BSD-2-Clause | обмен массивами с NumPy |
| rayon 1.12 | MIT OR Apache-2.0 | параллельные вычисления |
| las 0.11 | MIT | чтение LAS |
| laz 0.13 | Apache-2.0 | распаковка LAZ |
| byteorder 1.5 | Unlicense OR MIT | транзитивно |
| target-lexicon 0.13 | Apache-2.0 WITH LLVM-exception | только сборка |

`rust/expds_core/src/delaunay.rs` — копия delaunator 1.1.0 ([mourner/delaunator-rs](https://github.com/mourner/delaunator-rs), ISC, © 2018 Mapbox; текст лицензии — в заголовке файла). Изменены точная в f64 ориентация для целых координат и перенумерация вершин в порядке вставки; треугольники совпадают с оригиналом побитово (тест `fill.rs`). Сам крейт delaunator остаётся зависимостью для сеток больше 2^25 по стороне.

Остальные транзитивные крейты — MIT и/или Apache-2.0 (проверено по Cargo.toml в реестре 19.10.2026). Перепроверка: `cargo install cargo-license && cargo license` в `rust/expds_core`.

## Python

| Пакет | Лицензия | Назначение |
|---|---|---|
| numpy, scipy | BSD-3-Clause | вычисления |
| shapely | BSD-3-Clause (внутри GEOS — LGPL-2.1, динамическая библиотека) | геометрия |
| pyproj | MIT (внутри PROJ — MIT) | системы координат |
| rasterio | BSD-3-Clause (внутри GDAL — MIT) | растры |
| laspy, lazrs | BSD-2-Clause; lazrs — MIT [проверить по метаданным пакета] | чтение LAS/LAZ |
| torch | BSD-3-Clause | нейросети |
| segmentation-models-pytorch | MIT | U-Net |
| timm | Apache-2.0 | энкодеры |
| lightgbm | MIT | оценщик кандидатов |
| opencv-python-headless | Apache-2.0 | только тесты паритета |

**ultralytics (AGPL-3.0) в каркасе не используется.** По ответу организаторов AGPL допустим, но с проверкой архитектуры. MIT/Apache-стек эту проверку снимает.

## Предобученные веса

| Веса | Статус |
|---|---|
| ImageNet-веса энкодеров (torchvision/timm через smp) | **Разрешено 29.10 (по сообщению бота 1 от организаторов).** Код — BSD/Apache; набор ImageNet распространяется на условиях для некоммерческих исследований, поэтому вопрос и задавался. Запасной путь остаётся — `python -m expds.train --no-pretrained` |
| RF-DETR Nano, Small, Medium, Large (детекция) | Код и эти размеры — Apache-2.0 по таблице README [roboflow/rf-detr](https://github.com/roboflow/rf-detr) (arXiv:2511.09554). Основа — DINOv2, у нас уже записан как Apache-2.0 на код и веса после перелицензирования 2023 года. Сам чекпойнт доучен на COCO: аннотации COCO — CC BY 4.0, изображения — Flickr под разными лицензиями, тот же класс вопроса, что веса ImageNet. Было ли предобучение на Objects365 (у предшественника LW-DETR было), README не говорит; у Objects365, по сторонним каталогам, некоммерческие условия (официальный текст не проверен). Вопрос в письмо организаторам вместе с ImageNet. В сдачу не входит, пока нет письменного ответа о предобученных весах и честного табло |
| RF-DETR Atto, Femto, Pico, XL, 2XL (детекция) | **Не использовать:** в той же таблице лицензия PML 1.0, пакет `rfdetr_plus` |
| RF-DETR-Seg, все размеры таблицы | В таблице сегментации README стоит Apache-2.0, включая XL и 2XL. Брать файл весов только если карточка этого файла повторяет Apache-2.0. В сдачу не входит |
| ADAF (Zenodo 15848663) | **Не использовать:** CC BY-SA 4.0. ShareAlike конфликтует с передачей исключительных прав (п. 12.5) |
| DINOv3 | **Не использовать:** лицензия Meta требует распространять производные на её условиях и сохраняет права Meta. Это мешает передаче исключительных прав (п. 12.5) |
| SegFormer `mit_b*` в segmentation-models-pytorch | **Не использовать:** исходная лицензия NVIDIA на эти веса — только некоммерческое использование |

### Предобученные веса, реально использованные в пакете (bot-7 v3, 29.10.2026)

Поиск по репозиторию: `rg "encoder_weights|weights=|pretrained|from_pretrained|torch.hub|timm.create_model" expds scripts inference`.
На инференсе ничего не скачивается: сети строятся с `encoder_weights=None` (`infer_nn.py:75,78`), `_network(..., False)`
(`crop_model.py`), D-FINE — из локального каталога через `load_state_dict` (`sp_dfine.py`, без хаба).

| Веса | Где | Источник, ревизия | Лицензия | Данные предобучения | В пакете v3 |
|---|---|---|---|---|---|
| ResNet-18 ImageNet (инициализация энкодера 9 U-Net: li_r18_hard2, li_dev_r18, li_gor_mask_r18, li_fort_mask_r18, spor_focal_r18, spor_focal_wide_r18, spor_gor_r18, surf_arch_r18, ae_fort_mask_r18) | только обучение, `expds/train.py:497,500` (`encoder_weights="imagenet"`) | smp: HF `smp-hub/resnet18.imagenet` @ `3f2325ff978283d47aa6a1d6878ca20565622683` (smp ≥0.4); первоисточник `https://download.pytorch.org/models/resnet18-5c106cde.pth` (torchvision, тот же файл в старых smp) | код torchvision BSD-3-Clause, smp MIT; карточка HF: `license: other` (ссылка на torchvision) | ImageNet-1k | да — как инициализация, внутри доученных весов; разрешено 29.10 |
| sp_crop.pt (кроп-классификатор resnet18) | `expds/crop_model.py` | своё обучение, в метаданных `'pretrained': False` | — | только трейн КОЗ-3 | да |
| LightGBM (kurgan_lgbm, rerank_arch_lgbm, pixgbm/all) | `expds/` | своё обучение | lightgbm MIT | только трейн КОЗ-3 | да |
| D-FINE-L COCO (инициализация SpOr-детектора bot-9, 5 фолдов) | `expds/sp_dfine.py`, флаг `[sp_dfine] enabled=false` | HF `ustc-community/dfine-large-coco` @ `aa8f8211545e4cff5adf52ea512f9ac74a9824ec`; бэкбон HGNetV2-B4 (PaddleClas PPHGNetV2_B4_stage1) | Apache-2.0 | COCO (только COCO: варианты `*-obj365`/`*-obj2coco` — отдельные чекпойнты, не использовались); бэкбон — ImageNet (SSLD, по PaddleClas) | **нет** (флаг выкл., веса не в архиве) |
| DINOv2-S (гейт городищ bot-5) | `expds/gor_gate.py`, флаг `[gor_gate] enabled=false`, заглушка | HF `facebook/dinov2-small` @ `ed25f3a31f01632728cabb09d1542f84ab7b0056` | Apache-2.0 | LVD-142M (Meta) | **нет** (артефактов нет) |

Objects365 нигде не используется. Запрещённых весов из таблицы выше (DINOv3, SegFormer mit_b*, ADAF, RF-DETR PML) нет.

## Сторонние данные

Пока не используются. Если добавите, берите их только из первоисточника, с проверкой лицензии, и записывайте в таблицу ниже.

| Набор | Ссылка | Лицензия | Подтверждение | Где использован |
|---|---|---|---|---|
| | | | | |

Кандидаты: CC BY 4.0 / CC0 (геофизика Boviolles / Vieil-Évreux, Chactún, UCL) — проверить на страницах источников. Нельзя: CC BY-NC-SA (Harlow Hill, Yambol, Bayan Gol) и Archaeoscape (запрет перераспространения).
