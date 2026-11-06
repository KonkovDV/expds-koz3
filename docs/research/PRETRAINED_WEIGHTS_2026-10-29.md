# Предобученные веса для КОЗ №3: полный аудит после разрешения от 29.10 (COCO / ImageNet / DINOv2)

Автор: Grok Bot (исследователь), 29.10.2026, МСК. Статус: исследование, ничего не обучалось.
Не дублирует, а дополняет `koz3_shared/DETECTORS_SOTA.md` (28.10: таблицы детекторов, лицензии §8, план Kaggle на 13 дней). Здесь: что меняет разрешение 29.10, лицензии сверены заново по API, новые находки (ADAF, HighResCanopyHeight, RF-DETR-Seg, DEIMv2-NC, OlmoEarth), ТОП-5 по приросту на GPU-час и черновик письма организаторам.

**Обозначения.** [ф] — факт, источник указан. [оц] — оценка, проверена не была. Δ — изменение honest-10 к базе A = 0,3146.
**Как сверялись лицензии (29.10).** GitHub API `repos/*/license` (поле SPDX и текст LICENSE), карточки и теги HF через `huggingface.co/api/models`, Zenodo API, README. Если код и веса лицензированы по-разному, считается лицензия весов.

---
## 0. Коротко
1. В сдачу бесплатно попадает D-FINE-L COCO на SpOr: 5-фолдовый OOF уже дал **+0,0198** (П1–П5 ✓, `bot-9/DFINE.md` §full_rest). Раньше был «только замер», после 29.10 веса можно сдавать. Обучать ничего не нужно, осталось собрать пакет и измерить время на 4090.
2. Самый большой незакрытый резерв по GAP.md — **Or и Ae без D-FINE**: recall povr Or 0,13, povr Ae 0,14, arch Or 0,06, gor Or 21/69. Рецепт тот же, что на SpOr, с теми же весами COCO, ≈5–6 GPU-ч.
3. Для лидара фундамент-модели ДЗЗ **не нужны**. В Archaeoscape (arXiv 2412.05203) на рельефе ImageNet-CNN и PVTv2 сильнее Satlas, DINOv2, DOFA и ScaleMAE. Теперь можно ImageNet-инициализация (c3in/c5in) и PVTv2-b2.
4. Спросить организаторов стоит в первую очередь про: **Objects365** (D-FINE obj2coco +3,3 AP COCO, RF-DETR/RF-DETR-Seg), **ADAF** (Ирландия, ALS, курганы/ограды/рингфорты, CC-BY-SA-4.0), **SatlasPretrain Aerial** (NAIP 0,5–2 м, ODC-BY), **Meta HighResCanopyHeight SSL** (Maxar 0,5 м, Apache-2.0), **SAM 2.1** (SA-1B/SA-V), **ImageNet-21k/22k/12k**.
5. Для геофизики готовых весов, подходящих по домену и лицензии, нет. Сильнее всего физика + GBM. Дополнительно — синтетическое предобучение на своих прямых моделях (внешних весов не требует).

---
## 1. Что изменилось 29.10 и что уже сделано
| Уже сделано / в работе | Источник | Что даёт разрешение 29.10 |
|---|---|---|
| D-FINE-L COCO (HF `ustc-community/dfine-large-coco`), SpOr, 5 фолдов: A 0,3146 → 0,3344 | `bot-9/DFINE.md` строки 323–342 | COCO разрешён, можно в сдачу |
| ConvNeXt-T U-Net для лидара, c3/c5 × sc/in; ImageNet-варианты отложены «до письма» | `bot-9/CONVNEXT.md` стр. 121 | ImageNet разрешён, запускать c3in/c5in |
| DINOv2-S, фильтр городищ без лидара: +0,005 | `bot-5/NOLIDAR.md` §12 | DINOv2 B/L/g разрешены |
| RF-DETR (rfdetr 1.11.2), DINOv2-S | `bot-5/FOR_BOT9_RFDETR.md` | **Чекпойнты RF-DETR предобучены на Objects365**, см. §3. Без чекпойнта (только энкодер DINOv2) можно |
| Ae-классификатор курганов на DINOv2-S: AUC 0,672 < ручные признаки 0,699, отклонён | `bot-5/NOLIDAR.md` §13 | B/L — повтор с риском. Ниже предлагается детектор, а не классификатор |

---
## 2. Сверка лицензий (29.10)
| Веса / репозиторий | Лицензия весов | На чём обучены | Где проверено | Вердикт |
|---|---|---|---|---|
| D-FINE COCO (`ustc-community/dfine-{large,xlarge}-coco`), репозиторий Peterande/D-FINE | Apache-2.0 | COCO, бэкбон HGNetv2 (PaddleClas, ImageNet) | GitHub API: Apache-2.0; теги HF | ✅ разрешено (COCO) |
| D-FINE `*_obj2coco`, `*_obj365` | Apache-2.0, но README: «*may be subject to the Objects365 dataset terms*» | Objects365 → COCO | [README D-FINE](https://github.com/Peterande/D-FINE#objects365coco) | ❓ спросить |
| DEIM-D-FINE L/X COCO (Intellindust-AI-Lab/DEIM) | Apache-2.0 (файл LICENSE) | COCO | GitHub API, текст LICENSE | ✅ (чекпойнты лежат на Google Drive) |
| **DEIMv2** | **собственная, Non-Commercial** + бэкбон DINOv3 | — | текст LICENSE DEIMv2 | ⛔ |
| RT-DETRv2 r50/r101 (lyuwenyu/RT-DETR) | Apache-2.0 | COCO (есть и O365-версии) | GitHub API; HF | ✅ только COCO |
| RF-DETR N/S/M/L и Seg N…2XL | Apache-2.0 | **Objects365 (Seg — маски SAM2) → COCO** | [README](https://github.com/roboflow/rf-detr), [arXiv 2511.09554](https://arxiv.org/html/2511.09554v2) | ❓ спросить (O365 + SAM2) |
| RF-DETR A/F/P/XL/2XL (detection) | PML 1.0 | — | README | ⛔ |
| DINOv2 S/B/L/g, с регистрами и без (`facebook/dinov2-*`, timm `*_dinov2.lvd142m`) | Apache-2.0 | LVD-142M | GitHub API facebookresearch/dinov2, HF | ✅ |
| timm ImageNet-1k: convnext_*.fb_in1k, pvt_v2_b2/b3.in1k, efficientnetv2, resnet | Apache-2.0 | ImageNet-1k | HF, теги timm | ✅ |
| timm `*.fb_in22k_ft_in1k`, `*.in12k_ft_in1k`, caformer in22k | Apache-2.0 | **ImageNet-21k/22k/12k** | HF | ❓ уточнить, считается ли это «ImageNet» |
| hrnet_w32 (timm), swinv2 | MIT | ImageNet-1k | HF | ✅ |
| PVT / PVTv2 (whai362/PVT) | Apache-2.0 | ImageNet-1k | GitHub API | ✅ (закрывает вопрос из DETECTORS_SOTA) |
| SatlasPretrain (все чекпойнты, в т.ч. **Aerial_SwinB_SI/MI**) | **ODC-BY** (README: «Checkpoints are released under ODC-BY»); тег HF apache — неверный | NAIP и др. 0,5–2 м | [README satlaspretrain_models](https://github.com/allenai/satlaspretrain_models#readme), torchgeo `naip.csv` = ODC-BY | ❓ спросить (по смыслу как CC-BY) |
| **Meta HighResCanopyHeight SSL** (ViT-H 2,9 ГБ, ViT-H квант. 749 МБ, ViT-L квант. 400 МБ) | **Apache-2.0** (README: «code and model weights are released under the Apache License 2.0») | ≈18 млн снимков Maxar, метод DINOv2, фильтр «в основном растительность» | [README](https://github.com/facebookresearch/HighResCanopyHeight#license), GitHub API | ❓ спросить: это не «DINOv2-веса», а свой SSL-прогон |
| **ADAF** (Faster R-CNN + HRNet: barrow / enclosure / ringfort / все объекты) | **CC-BY-SA-4.0** (Zenodo); код Apache-2.0 | ALS Ирландии, SLRM, 512 px | [Zenodo 15848663](https://zenodo.org/records/15848663), [EarthObservation/adaf](https://github.com/EarthObservation/adaf) | ❓ спросить (ShareAlike — не из списка) |
| SAM 2.1 | Apache-2.0 | SA-1B + SA-V | HF | ❓ спросить (данные SA-1B) |
| Prithvi-EO-2.0, Clay, TerraMind, Panopticon | Apache-2.0 | Sentinel/Landsat/HLS 10–30 м | GitHub API, HF, torchgeo `agnostic.csv` | ❓ разрешимо, но пользы мало |
| Galileo, AnySat, CROMA, DEO, DOFA (код MIT / веса CC-BY-4.0), SSL4EO-S12, Copernicus-FM | MIT / CC-BY-4.0 | 10 м и грубее (AnySat захватывает и аэро 0,2 м) | GitHub API, torchgeo | ❓ пользы мало |
| DeepAndes (ViT-L, DINOv2 на 8-канальных снимках Анд) | MIT | WorldView/спутники Анд | [geopacha/DeepAndes](https://github.com/geopacha/DeepAndes) | ❓ мультиспектр, у нас RGB → мало пользы |
| RemoteCLIP | код Apache-2.0, у весов на HF нет тега; данные — DOTA/DIOR и др. (исследовательские) | — | GitHub API, HF | ⛔ не брать |
| GeoRSCLIP | «cc» без версии | — | HF | ⛔ |
| MTP / RSP (ViTAE) | код MIT; веса обучены на SAMRS/MillionAID (данные DOTA/DIOR/FAIR1M — только для исследований) | — | GitHub API | ⛔ (лицензии данных) |
| SkySense / SkySense V2 | LICENSE нет / репозиторий 404 | — | GitHub API | ⛔ |
| ScaleMAE, Mask2Former Swin (torchgeo), Restor TCD | CC-BY-NC-4.0 | — | torchgeo `agnostic.csv`, `aerial.csv` | ⛔ |
| OlmoEarth | OlmoEarth Artifact License (своя) | — | текст LICENSE | ⛔ |
| BudjBim DINO-CV (SSL на лидарных SLRM/VAT — ровно наш домен) | CC-BY-NC-4.0 | — | текст LICENSE | ⛔ |
| HF `kardosdrur/burial-mounds-yolov8m*` | CC-BY-NC + ultralytics | — | HF | ⛔ |
| C-RADIOv3, DINOv3(-SAT), SAM 3, SegFormer mit_*, ultralytics | NVIDIA OML / Meta / AGPL | — | DETECTORS_SOTA §8 | ⛔ |
| DAI AnomalyExtractor (магнитограммы) | **GPL-3.0** | — | GitHub API | ⛔ код; идею (порог ±5 нТл → векторизация) реализовать самим |

---
## 3. Кандидаты по типам данных
Колонки: модель | веса | источник | лицензия | размер | 4090 инференс [оц] | T4 обучение | ожидаемый прирост (обоснование) | риски.
Скорость на 4090 [оц]: от T4 TensorRT-fp16 из README × ~0,4 (4090 примерно в 2,5–3 раза быстрее T4 в fp16) × 1,5–2 (PyTorch без TRT). Число окон у бота 9: SpOr ≈ 3256 окон по 640 px на 5 фолдов (`DFINE.md` стр. 273, 315).

### 3.1 Ортофото / спутник / аэро (Or, SpOr, Ae)
| Модель | Веса | Источник | Лицензия | Размер | 4090 [оц] | T4 обучение | Прирост | Риски |
|---|---|---|---|---|---|---|---|---|
| **D-FINE-L COCO → SpOr** (уже есть) | ustc-community/dfine-large-coco @aa8f8211 | HF | Apache, COCO ✅ | 31 M, ≈125 МБ fp32 | ≈5–8 мс/окно → 1–2 мин на тест [оц] | сделано (5 фолдов) | [ф] **+0,0198** (`DFINE.md`) | для сдачи нужна модель на всех данных (≈2 ч T4) |
| **D-FINE-L COCO → Or + Ae** | то же | HF | ✅ | 31 M | +1–3 мин [оц] | ≈1,6–1,9 ч на пару фолдов × 3 ≈ **5–6 ч** | [оц] +0,005…+0,015: GAP — povr Or/Ae recall 0,13/0,14, arch Or 0,06; на SpOr D-FINE 0,276 против стека 0,166 | Ae/Or разнородны по разрешению; данных меньше, чем в SpOr |
| DEIM-D-FINE-L COCO | Google Drive DEIM | [DEIM](https://github.com/Intellindust-AI-Lab/DEIM) | Apache ✅ | 31 M | как D-FINE | как D-FINE (нужен код DEIM, не HF) | [ф] COCO 54,7 против 54,0 (+0,7 AP); [оц] Δ ≤ +0,002 | чекпойнт не в HF-формате, нужна конвертация |
| D-FINE-X COCO | ustc-community/dfine-xlarge-coco | HF | Apache ✅ | 62 M | ≈1,6× L | ≈1,6× L | [ф] COCO 55,8 против 54,0; [оц] +0,002…0,005 к L | VRAM на T4, bs 4 |
| D-FINE-L/X obj2coco | `dfine_l_obj2coco_e25.pth` | [README](https://github.com/Peterande/D-FINE) | ❓ O365 | 31/62 M | как L/X | как L/X | [ф] COCO 57,3 / 59,3 (+3,3 / +3,5 к COCO-only); авторы: «Objects365 pretrained … best generalization»; [оц] +0,003…0,008 | нужно разрешение |
| RF-DETR-M/L (det) | rfdetr 1.11.2 | [README](https://github.com/roboflow/rf-detr) | ❓ O365 | 34 M | [ф] T4 TRT 4,4 / 6,8 мс | ≈1,5–2 ч на фолд-пару [оц] | [ф] COCO 54,7 / 56,5; RF100-VL у них лучший; [оц] сопоставимо с obj2coco | без чекпойнта (DINOv2 только в энкодере) в разы теряет — [оц] |
| **RF-DETR-Seg-M/L** (маски → полигоны arch/object_poly) | rfdetr | README | ❓ O365 + SAM2 | 36 M | [ф] T4 5,9 / 8,8 мс | ≈2 ч [оц] | [оц] arch/object_poly — классы с полигонами, где рамка → эллипс теряет форму | нужно разрешение |
| RT-DETRv2-r50 COCO | HF PekingU/rtdetr_v2_r50vd | HF | Apache ✅ | 42 M | ≈ D-FINE | ≈ D-FINE | [оц] ≤ D-FINE (COCO 53,4 у DEIM-версии) | дубль |
| **DINOv2-B/L-reg верификатор** (городища без лидара 384/768 м, arch Or) | facebook/dinov2-with-registers-{base,large} | HF | Apache ✅ | 86 M / 304 M | 10–40 мс/виньетка [оц] | признаки ≈0,3–1 ч, голова на CPU | [оц] +0,001…0,004 к текущему +0,005 (S → B/L обычно +1–3 п. linear-probe) | §13: DINOv2-S на Ae-курганах проиграл ручным признакам — для курганов не повторять |
| SatlasPretrain Aerial Swin-v2-B (SI) + FPN как бэкбон D-FINE или верификатора | aerial_swinb_si.pth | [README](https://github.com/allenai/satlaspretrain_models) | ❓ ODC-BY | 88 M | ≈20 мс/окно 512 [оц] | 3–4 ч [оц] | [оц] домен 0,5–2 м совпадает с Or/Ae; в Archaeoscape Satlas Swin-B 48,4 < IN1k 50,7 — но это **лидар**, для RGB ожидаем лучше | у EN-конкурентов не валидировано (NOLIDAR §13) |
| Meta HRCH SSL ViT-L (квант. 400 МБ) / ViT-H | s3://dataforgood-fb-data/forests/v1/models | [README](https://github.com/facebookresearch/HighResCanopyHeight) | ❓ Apache, Maxar | 300 M / 630 M | ViT-L ≈40 мс/виньетка 256 [оц] | только признаки ≈1 ч | [оц] сильнее DINOv2 на 0,5 м спутнике (тот же метод, домен — Maxar); кандидат на верификатор городищ и курганов Ae | ViT-H 2,9 ГБ > 2 ГБ архива, только ViT-L; предобучение на «растительности» |
| Prithvi / Clay / TerraMind / DOFA / Galileo / Copernicus-FM | — | §2 | ✅/❓ | — | — | — | [ф] Archaeoscape: DOFA 33,6, ScaleMAE 22,8 IoU — хуже ImageNet; домен 10–30 м | не тратить GPU |

### 3.2 Лидар (ЦМР 0,1–0,25 м, каналы рельефа)
| Модель | Веса | Источник | Лицензия | Размер | 4090 [оц] | T4 обучение | Прирост | Риски |
|---|---|---|---|---|---|---|---|---|
| **ConvNeXt-T U-Net, ImageNet (c3in/c5in)** | timm convnext_tiny.fb_in1k | HF timm | Apache ✅ | 32 M | [оц] 8–16 мин на 3 модели (BENCH_T4 §94, T4) → на 4090 ≈ 3–6 мин | 12 000 ит × 0,33 с ≈ 1,1 ч/фолд → ≈5,5 ч на 5 фолдов | [ф] Archaeoscape: U-Net IN1k 50,7 против random 46,1 (+4,6 IoU); [оц] Δ +0,003…0,010 к sc | c3sc-f01 дал −0,0105 из-за переноса порога между 2 семьями (CONVNEXT.md 187–192) — сравнивать только на всех 44 семьях |
| **PVTv2-b2 U-Net/FPN, ImageNet** | timm pvt_v2_b2.in1k | HF timm, [PVT](https://github.com/whai362/PVT) | Apache ✅ | 25 M | ≈ ConvNeXt-T [оц] | ≈1,3× ConvNeXt-T [оц] → ≈7 ч | [ф] Archaeoscape: PVTv2 **53,0** — лучший результат (IoU курганов) против 50,7 U-Net IN1k | smp 0.6 поддерживает `tu-pvt_v2_b2`; на T4 проверить VRAM |
| HRNet-W32 U-Net, ImageNet | timm hrnet_w32 | HF | MIT ✅ | 41 M | 1,5× [оц] | ≈1,5× | [оц] ≈ ConvNeXt; ADAF выбрал HRNet для сегментации курганов на SLRM | дубль, только ради ансамбля |
| **D-FINE-L COCO на рельефе** (3 канала: SLRM/VAT, уклон, отмывка) — рамки → круги для povr/tselye, полигоны gor/fort | ustc-community/dfine-large-coco | HF | ✅ | 31 M | ≈2–4 мин [оц] | ≈6 ч (5 фолдов) [оц] | [оц] +0,003…0,010: детектор лучше выделяет отдельные курганы, чем сегментация + постобработка; прецедент ADAF — Faster R-CNN на SLRM для barrows | 0,1–0,25 м → привести к 0,25–0,5 м, чтобы курган 10–30 м входил в окно |
| ADAF Faster R-CNN / HRNet (barrow, enclosure, ringfort) | Zenodo 15848663 | [Zenodo](https://zenodo.org/records/15848663) | ❓ CC-BY-SA-4.0 | tar 518 МБ (OD) / 862 МБ (HRNet), с оптимизатором [оц] | ≈ Faster R-CNN R50 [оц] | дообучение 2–4 ч | [оц] курганы (barrows) и ограды/рингфорты ≈ городища — ближайший по домену чекпойнт; лучше всего как старт для дообучения | ShareAlike; SLRM Ирландии ≠ наш DEM; нужен фреймворк AiTLAS |
| SAM 2.1 (zero-shot полигоны городищ по виньетке) | facebook/sam2.1-hiera-* | HF, код Landauer [CC0](https://github.com/juergenlandauer/FoundationModelsArchaeology) | ❓ данные SA-1B | 39–224 M | ≈50 мс/виньетка | 0 | [оц] улучшает форму полигона gor/fort (покрытие ≥50%) | нужно разрешение |
| Satlas / DINOv2 / DOFA / ScaleMAE на рельефе | — | Archaeoscape | — | — | — | — | [ф] Swin-B Satlas 48,4, DINOv2 ViT-S 40,9, DOFA 33,6, ScaleMAE 22,8 — **хуже** ImageNet-CNN | не тратить GPU |

### 3.3 Геофизика (магнитометрия)
| Модель | Веса | Источник | Лицензия | Размер | 4090 [оц] | T4 | Прирост | Риски |
|---|---|---|---|---|---|---|---|---|
| Текущий пиксельный генератор + LightGBM | — | бот 5/10 | своё | — | секунды | 0 | база (Geo теряет 0,131, GAP) | — |
| EfficientNet/ConvNeXt-T U-Net, ImageNet, на 3 каналах (клип нТл, \|∇\|, аналитический сигнал) | timm *.in1k | HF | ✅ | 20–30 M | < 1 мин | ≈2 ч [оц] | [оц] 0…+0,004: данных мало, ImageNet помогает в основном со сходимостью | object_poly Geo recall 0,07 → в основном FN кандидатов, а не классификация |
| **Предобучение на синтетике** (прямая модель: призмы, цилиндры, диполи) | свои | идея: [Digital Geophysical Eye](https://doi.org/10.21203/rs.3.rs-10207513/v1) (CNN 90,7% на синтетике); [C&G 2025, сегментация магнитного поля](https://doi.org/10.1016/j.cageo.2024.105847) | своё ✅ | — | — | 1–2 ч [оц] | [оц] 0…+0,005 | синтетика ≠ реальность; внешних весов нет |
| D-FINE-L COCO на рендере магнитограмм | HF | — | ✅ | 31 M | < 1 мин | ≈2 ч | [оц] неизвестно; аномалии — пятна и линии, COCO даёт мало | низкий приоритет |
| Готовые веса для магнитометрии | — | поиск HF/GitHub 29.10: подходящих открытых весов нет | — | — | — | — | — | — |

---
## 4. ТОП-5 по ожидаемому приросту на GPU-час (рецепты для бота 9)
Ранжировано по [оц] Δ / GPU-ч T4. Все варианты проходят П1–П7 (`EXPERIMENTS.md`). Каждый должен быть вписан в ≤ 1200 с на 4090.

**1. D-FINE-L COCO (SpOr) — в сдачу.** [ф] Δ +0,0198 за 0 новых GPU-ч на подбор (финальная модель на всех SpOr ≈ 2 ч T4).
- Репозиторий: `koz3_shared/work/kaggle/dfine_spor` (det_train_hf.py). Чекпойнт: `ustc-community/dfine-large-coco` @ `aa8f8211545e4cff5adf52ea512f9ac74a9824ec`.
- Режим как в 5-фолдовом прогоне: 0,5 м/пикс, окно 640, шаг 160 при инференсе, 6000 ит, bs 8, AdamW 1e-4 (бэкбон ×0,1), wd 1e-4, OneCycle 0,05, clip 0,1, fp16. Аугментации: 60% окон с объектами, сдвиг ±1/3, вес класса 1/√n (потолок 4), повороты 90° + отражения.
- Финал: обучить на всех 5 фолдах (full-data) 6000–8000 ит. В пакет — fp16 `weights_only` (≈62 МБ). Вариант C или E; `threshold_group=sp_dfine`; старые sp-группы не выбрасывать (R0 −0,0064).
- Проверка: время всего SpOr-теста на 4090 и бюджет 1200 с.

**2. D-FINE-L COCO на Or + Ae (новая ветка `or_dfine`, `ae_dfine`).** [оц] Δ +0,005…+0,015 за ≈5–6 GPU-ч.
- Код тот же, меняются только `markup_type` ∈ {Or, Ae} и фолды `folds_family_k5.json`. Обучать одной моделью Or+Ae с флагом-доменом в сэмплере. 0,5 м/пикс (Ae, если грубее, — нативно, без апсемпла выше 2×).
- 640 px, 6000 ит, bs 8, те же гиперпараметры. Аугментации: + ColorJitter (яркость/контраст 0,2, насыщенность 0,2, оттенок 0,02), + случайный масштаб 0,8–1,25 (разное разрешение Ae), + GaussianBlur p 0,2.
- 3 полосы по 2 фолда на T4×2 (≈1,7 ч каждая). Затем слияние и `family_curves.py` с отдельными threshold_group.
- Ожидаемые классы: povr, gor (Or 21/69), arch.

**3. Лидар: ImageNet вместо «с нуля» + PVTv2-b2.** [оц] Δ +0,003…+0,010 за ≈5,5–7 GPU-ч на вариант.
- Репозиторий `bot-9/kaggle/convnext_lidar`. Варианты: (а) `c3in`/`c5in` = `smp.Unet("tu-convnext_tiny", encoder_weights` из `convnext_tiny.fb_in1k`); (б) `smp.Unet("tu-pvt_v2_b2")` или FPN с `pvt_v2_b2.in1k`.
- Для 5 каналов первый слой: веса RGB усреднить и повторить на 5 каналов × 3/5 (стандартная инициализация timm `in_chans=5`).
- 12 000 ит, тайл 512 при 0,25 м (как сейчас), lr энкодера 1e-4, декодера 1e-3, AdamW, cosine, fp16. Аугментации: D4 (повороты 90° + отражения), сдвиг высоты ±, масштаб 0,8–1,2. Без цветовых аугментаций рельефа, кроме случайного азимута отмывки.
- **Важно:** сравнивать in и sc только после того, как модель покрывает все 44 лидарные семьи (иначе метрику портит перенос порога, CONVNEXT.md 187–192).

**4. DINOv2-B/L-reg для верификаторов городищ без лидара и arch Or.** [оц] Δ +0,001…+0,004 за ≈1 GPU-ч (признаки один раз, голова LGBM/логрег на CPU).
- `facebook/dinov2-with-registers-base` (и `-large` для сравнения), виньетки 384/768 м как в NOLIDAR §12. Вход 518 px (37×37 патчей), CLS + среднее по патчам, TTA D4 (усреднение признаков).
- Голова — та же, что у текущего clean gate. Сравнение S/B/L по тем же фолдам.
- Для 4090 — ViT-B fp16, ≈10 мс/виньетка [оц]. Курганы Ae на DINOv2 не повторять (NOLIDAR §13).

**5. D-FINE-L COCO на рельефе лидара (детектор курганов и городищ).** [оц] Δ +0,003…+0,010 за ≈6 GPU-ч.
- Код `dfine_spor` с входом из 3 каналов рельефа вместо RGB: (R) SLRM/LRM 20 м, клип ±0,5 м; (G) уклон 0–45°; (B) многонаправленная отмывка. Ресемпл к 0,25 м: курган ⌀10–30 м → 40–120 px; городища — отдельный проход при 1 м.
- 640 px, 6000 ит, bs 8, аугментации D4 + масштаб 0,8–1,25, без ColorJitter. Классы povr, tselye, gor, fort.
- Рамка → круг для povr/tselye (центр ≤ 2 м: точность центра у D-FINE высокая — FDR-регрессия) [оц].
- Вход в стек отдельной threshold_group `li_dfine`, ансамбль с ConvNeXt через LGBM-стек.

Дальше (при разрешении организаторов, ожидаемая выгода выше): заменить чекпойнт в п.1/2/5 на **D-FINE-L obj2coco e25** (+3,3 AP COCO) или RF-DETR-L/Seg-L; в лидаре стартовать от ADAF; на Ae/Or — Satlas Aerial SwinB как бэкбон верификатора. D-FINE-X COCO против L — только если останется GPU (≈1,6× по времени).

---
## 5. Что спросить у организаторов отдельно (черновик письма)
Порядок — по ожидаемой пользе.

1. **Objects365** — чекпойнты, предобученные на Objects365 и дообученные на COCO: D-FINE `*_obj2coco` и `*_obj365` (Apache-2.0, github.com/Peterande/D-FINE), DEIM-D-FINE-X O365, RF-DETR N/S/M/L и RF-DETR-Seg (Apache-2.0, github.com/roboflow/rf-detr; маски для предобучения сделаны SAM2).
2. **ImageNet-21k/22k/12k**: считается ли это частью разрешённого «ImageNet»? (timm `*.fb_in22k_ft_in1k`, `*.in12k_ft_in1k`; HGNetv2-бэкбон D-FINE — PaddleClas SSLD.)
3. **ADAF** — веса Faster R-CNN / HRNet для курганов, оград и рингфортов на ALS Ирландии (zenodo.org/records/15848663, CC-BY-SA-4.0; код Apache-2.0).
4. **SatlasPretrain** (allenai), в т.ч. Aerial Swin-v2-B (NAIP 0,5–2 м), лицензия ODC-BY.
5. **Meta HighResCanopyHeight SSL-бэкбон** (ViT-L/H, метод DINOv2, ≈18 млн снимков Maxar), Apache-2.0 — относится ли он к разрешённым «весам DINOv2»?
6. **SAM 2.1** (Apache-2.0, данные SA-1B / SA-V) — для уточнения полигонов.
7. По желанию (польза ниже): Prithvi-EO-2.0, Clay, TerraMind, Panopticon (Apache-2.0); Galileo, AnySat (MIT); DOFA, SSL4EO-S12, Copernicus-FM (CC-BY-4.0).

**Черновик (рус.):**
> Здравствуйте! Команда StrataTrace, КОЗ №3. Спасибо за разрешение от 29.10 использовать веса, предобученные на COCO, ImageNet и DINOv2. Уточните, пожалуйста, допустимы ли также следующие открытые веса (все под разрешительными лицензиями, обучение и инференс — только локально):
> 1) детекторы, предобученные на Objects365 и дообученные на COCO: D-FINE obj2coco/obj365 (Apache-2.0), RF-DETR N/S/M/L и RF-DETR-Seg (Apache-2.0; маски для предобучения получены SAM2);
> 2) ImageNet-21k/22k/12k (timm *_in22k_ft_in1k, *_in12k_ft_in1k) — входят ли они в «ImageNet»;
> 3) веса ADAF (Faster R-CNN/HRNet по курганам и оградам на лидаре Ирландии), Zenodo 15848663, CC-BY-SA-4.0;
> 4) SatlasPretrain (Allen AI), в т.ч. модели для аэрофотоснимков 0,5–2 м, ODC-BY;
> 5) SSL-бэкбон Meta HighResCanopyHeight (обучен методом DINOv2 на спутниковых снимках Maxar), Apache-2.0;
> 6) SAM 2.1 (Apache-2.0, обучен на SA-1B/SA-V);
> 7) геофундамент-модели Prithvi-EO-2.0, Clay, TerraMind (Apache-2.0), Galileo (MIT), DOFA, SSL4EO-S12 (CC-BY-4.0).
> Если разрешение возможно только для части списка, просим в первую очередь ответить по п.1–3. Спасибо!

Отдельно сохранить в репозитории письменную копию разрешения от 29.10 (COCO/ImageNet/DINOv2): в `docs/` его пока нет, а RED_TEAM.md отмечает, что ответа по ImageNet нет.

---
## 6. Исключено (не тратить время)
DEIMv2 (NC + DINOv3), DINOv3 / DINOv3-SAT, SAM 3, C-RADIOv3, SegFormer mit_*, ultralytics/YOLO (AGPL), RF-DETR A/F/P/XL/2XL (PML), ScaleMAE / ConvNeXt-V2 FCMAE / Hiera (NC), SkySense (без лицензии), RemoteCLIP / GeoRSCLIP / MTP / RSP (неясные данные или лицензия), BudjBim DINO-CV (NC), OlmoEarth (своя лицензия), kardosdrur burial-mounds (NC + AGPL), код DAI AnomalyExtractor (GPL-3), код сторонний репозиторий (нет LICENSE).

## 7. Источники
- Archaeoscape: https://arxiv.org/abs/2412.05203 ; код (Apache-2.0 по GitHub API, бейдж README — MIT; раздел весов «Coming soon»): https://github.com/archaeoscape-ai/archaeoscape
- D-FINE: https://github.com/Peterande/D-FINE ; DEIM: https://github.com/Intellindust-AI-Lab/DEIM ; DEIMv2 LICENSE: https://github.com/Intellindust-AI-Lab/DEIMv2
- RF-DETR: https://github.com/roboflow/rf-detr ; https://arxiv.org/abs/2511.09554
- SatlasPretrain: https://github.com/allenai/satlaspretrain_models ; torchgeo, таблицы весов с лицензиями: https://github.com/torchgeo/torchgeo/tree/main/docs/api/weights
- HighResCanopyHeight: https://github.com/facebookresearch/HighResCanopyHeight ; https://arxiv.org/abs/2304.07213
- ADAF: https://zenodo.org/records/15848663 ; https://github.com/EarthObservation/adaf
- Landauer 2025, Foundation Models in Archaeology (CC0): https://github.com/juergenlandauer/FoundationModelsArchaeology
- DINO-CV (BudjBim, NC): https://github.com/MLinArcheaomatics/BudjBimStoneWall-DINO-CV ; https://doi.org/10.1016/j.srs.2026.100372
- DeepAndes: https://github.com/geopacha/DeepAndes
- Магнитометрия: https://doi.org/10.1016/j.cageo.2024.105847 ; https://doi.org/10.21203/rs.3.rs-10207513/v1 ; https://www.nature.com/articles/s41598-025-05539-3 ; https://github.com/dainst/AnomalyExtractor
- OpenAI to Z (победитель Archaios; деталей метода нет): https://www.kaggle.com/competitions/openai-to-z-challenge/writeups/archaios
- Внутренние: koz3_shared/{DETECTORS_SOTA,AUDIT,EXPERIMENTS}.md, bot-4/GAP.md, bot-5/{NOLIDAR,FOR_BOT9_RFDETR}.md, bot-9/{DFINE,CONVNEXT,BENCH_T4}.md
