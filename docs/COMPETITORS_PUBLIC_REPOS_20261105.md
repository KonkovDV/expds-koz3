# Публичные репозитории конкурентов КОЗ-3 — копия 05.11.2026 (git clone --depth 1)

Это только публичный код с GitHub. Docker-образы команд не скачивались: веса в них — их отправленные решения, берём только присланные самими командами (см. docs/THIRD_PARTY.md в expds-koz3). Разбор приёмов — docs/COMPETITORS_ATOMIC.md, docs/plans/PLAN_TO_061_20261105.md.

| Репозиторий (коммит) | Команда / место | Что ценного | Веса в репо | Лицензия |
|---|---|---|---|---|
| EhimenNathan/EXpedition (d434e56) | Сахалинский центр ИИ (EN), публичный LB v20 0,5019 | **Самое ценное.** Полный код обучения и инференса всех модальностей. Спутник: 5–6 моделей D-FINE + слияние на уровне участка (`koz3/sp_fuse.py`, порог 0,25, match 8 м), гейт сцены городищ ConvNeXt (`koz3/scene_cls.py`, AUC 0,89), приор кадра только для больших сцен, «хедж» повреждённый/целый 0,4. Аэрофото: FCOS фортов (`koz3/tv_det_train.py`, `ae_det_train.py`). Ортофото: D-FINE архитектуры ≥ 0,5. Слияние лидар→спутник α 0,1 (`koz3/fuse.py`). `docs/EXPERIMENTS.md` — журнал с проверками на LB: спутник у них .44/.64 против нашего .47/.27 | нет | нет |
| Strekalov/arhipelag (05.11) | **GigaFlex** (3-е КОЗ-3, 0,587) | Их решение конкурса по снимкам с дронов: рецепт обучения ultralytics (imgsz 1280, AdamW, box 29,7, сильные hsv/bgr/mixup/cutmix, degrees 35) и инференс в родном разрешении. Разбор — `docs/plans/OPTICS_DETECTOR_PLAN_20261105.md` §3.9. Отдельный архив `comp_gh_arhipelag.tar.gz` | нет | нет |
| Strekalov/ultralytics (b5ccd6d) | GigaFlex, 3-е, 0,587 (автор образов strekalov/*) | Форк ultralytics. Изменения GigaFlex надо искать сравнением с upstream; HEAD выглядит как upstream | нет | AGPL-3.0 (ultralytics) |
| yakirill99/ds-expedition (e3a803b, develop) | — | Пайплайн целиком: центры полигонов, деление компонент по пикам, заливка nodata (у нас всё это есть) | нет | нет |
| vaitter/koz3-build (dfdafaa) | — | YOLO-seg по отмывке лидара (`solution.py`, `hillshade_lib.py`) и **3 файла весов** `best.pt`, `best_e2.pt`, `best_e3.pt` (20–40 МБ), выложены самим автором | **да** | нет (ultralytics AGPL) |
| ObuObuHub/tumulus-lidar-detector (70ea999) | не КОЗ | Фильтры «двойников» курганов лидара: линейность и округлость контура (AUC 0,94). Это наш пункт B4: снижать FP лидара (на LB 408 ложных) | — | **MIT** — код можно брать |
| maratgaliulin/EXpedition (1941e05) | — | Копия или ранняя версия EN | нет | нет |
| vdurnov/xview2_1st_place_solution (8c70776) | xView2, 1-е (не КОЗ) | TTA отражениями у U-Net, сиамские U-Net | — | **MIT** |
| Iskatel-potenzialov/gis (79e613b) | КОЗ-1 | идеи | — | **MIT** |
| L0thlorien/ds-expedition, Egor-git-cloud/ds-expedition, KoshelevDV/ds-expedition-infra | — | каркасы и инфраструктура, приёмов нет | — | — |

## Docker Hub (метаданные слоёв, образы не скачивались)
- strekalov/koz3-submission, strekalov/rt-core (GigaFlex, 3-е, 0,587): ultralytics + yolo26n, RT-DETR.
- evgps/ds3v2-koz3 (k3063w/k3064): layer_sat 1,8 ГБ, layer_sat56nom 1,6 ГБ, layer_li 0,4 ГБ, layer_ae56 0,07 ГБ — «sealed weights».
- huntertomo/koz3-inference v3.3-d: YOLO на каждую модальность, SAM, отдельные модели на визуализации RVT.
- alek231/expedition-twostage: YOLO + timm/smp, две ступени. datomi79/expedition-ds: Point Transformer V3 по облаку LAS.
- andreiorlov/archaeo-solution-v1: Prithvi-EO-2.0-300M. bizziberry/koz3:seg1: yolo26n.
