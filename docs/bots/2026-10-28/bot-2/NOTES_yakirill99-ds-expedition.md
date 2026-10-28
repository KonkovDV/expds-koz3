# NOTES: yakirill99-ds-expedition (bot-2, 28.10.2026)

Источник: `/workspace/koz3_shared/competitors/yakirill99-ds-expedition`. Анализ только чтением, ничего не запускалось.
Пути ниже даны относительно корня репо. «Оценка» означает экспертное суждение без замера.

## 0. Главное: это каркас для КОЗ №3, балл 0,5–0,6 из него получить нельзя
- Клон неглубокий, в нём один коммит `90a45f6` от 2026-10-28 18:23:31 +0300: «этап A — GT-разметка 006, CRS-арбитраж по UTM.json, смок load_site (#36)». Соревнование то же: КОЗ №3 (`AGENTS.md:11`). Официальная метрика лежит вербатим в `vendor/organizers/compute_metrics.py`: без учёта хвостовых пробелов она совпадает с `/workspace/expds-koz3/materials/compute_metrics.py`, в обоих файлах 1202 строки, версия koz3-1.5.0 (`:192`).
- **На реальных данных модель не обучалась.** В `configs/dataset_real.yaml:107-108` списки `train_sites: []` и `val_sites: []` пустые, а `train_from_config` с пустым train сразу падает (`src/expds/train.py:314-315`). Реальных участков описано два: 013 и 048 (`configs/dataset_real.yaml:153-157`). Разметку 006 только начали импортировать (`scripts/import_labels.py:23-28`).
- Единственные заявленные цифры получены на синтетике. Это 3 тайла 1024², train site_1-2, val site_3: oracle F1 1,00, U-Net val F1 0,71 (p 0,62, r 0,84) (`docs/SKELETON.md:59-62`), и ещё «0,63 вместо 0,94» в комментарии к лоссу (`configs/dataset_real.yaml:66-68`). Метрика здесь своя, не официальная (§6).
- **Вывод репо официальная метрика не засчитает:**
  - класс пишется в свойство `class`, уверенность в `score` (`src/expds/io/geojson.py:37-38`, `write_detections` `:298-345`, свойства `:331-336`);
  - значения класса — `mound | rampart | ditch` (`configs/dataset_real.yaml:45`, `scripts/import_labels.py:14-16,48-59`);
  - официальная метрика читает `props.get("class_name", "unknown")` и отбрасывает всё, что не входит в 7 классов (`materials/compute_metrics.py:277-279,827-829`), а `confidence` по умолчанию берёт равным 1,0 (`:340-343`).
  - В таком виде сабмит получит ≈0 по всем классам. Это вывод из чтения кода, не замер.
- В OSINT репо найдено поиском по имени `ds-expedition` (`/workspace/koz3_shared/osint/code2.txt:120-121`). С баллом в рейтинге оно там не связано.

## 1. Модель, входы, предобработка
- 19 каналов: dtm, slope, 4 hillshade, slrm_sigma_8, pos/neg openness, SVF, curvature, TPI5, TRI3, point_density, mean_intensity, rgb, mag (`configs/dataset_real.yaml:18-37`). Нормализация `site_robust`, то есть по участку (`:39`).
- Из LAS строится DTM 1 м, фильтр земли пресет `soft`, который «сохраняет курганы» (`src/expds/features/README.md:32`, `src/expds/features/ground.py:8`). Рельеф: SLRM σ 2/8/32, openness r10 по 16 направлениям (конфиги dataset_006/013, см. `src/expds/features/relief.py:137,243,316,347`).
- U-Net resnet34 (smp) без предобучения: `encoder_weights=None` (`src/expds/models/factory.py:54-56`, `configs/dataset_real.yaml:56-58`). Последний bias инициализируется prior 0,01 (`factory.py:61-69`).
- Две головы на класс: seg и heatmap центров, выход 2K (`factory.py:15,40-41,72-77`).
- Тайлы 512, перекрытие 128, pos_fraction 0,7 через WeightedRandomSampler (`configs/dataset_real.yaml:49-52`, `src/expds/tiles/dataset.py:232-243`, `train.py:214-222`). Сшивка окном Ханна (`configs/dataset_real.yaml:72-74`).

## 2. Обучение
- AdamW lr 1e-3, wd 1e-4, cosine, AMP, 10 эпох, bs 8 (`src/expds/train.py:51-67,232-235`).
- Лосс: seg = 0,5·Dice + 0,5·focal(α 0,25, γ 2), hm = soft-focal (QFL) по гауссиане σ 3 px, нормированный на max(масса, hm_min_norm=2000) (`src/expds/models/koz_loss.py:14-24,66-85`, `configs/dataset_real.yaml:46,61-69`). hm-вес вводится разогревом (`train.py:255-257`).
- Аугментации: hflip/vflip/rot90 с p 0,5 и channel dropout 0,1 (`src/expds/tiles/augment.py:37-40`).
- Цели: Polygon растеризуется в маску, Point — в гауссов heatmap (`src/expds/targets/rasterize.py:3-5,73-176`).
- **Подклассы курганов сливаются:** «курганы_поврежденные» превращаются в `mound` (`scripts/import_labels.py:21,53,64,120-126`). При этом в их же чек-листе записано, что курганы дают 30+20 = половину метрики (`README-data-registry.md:45-46,95`). Целые и повреждённые модель различать не учится.
- Перед обучением считается oracle: таргеты прогоняются через тот же post и метрику (`train.py:131-143,244-251`). Чекпоинт выбирается по val F1 (`train.py:280-295`).

## 3. Постобработка
- seg: порог 0,5, 8-связные компоненты, субпиксельный контур, min_area 16 px, simplify 0,5 px; score = средняя вероятность (`src/expds/post/vectorize.py:1-14,34-46`, `:161-191`; конфиг `configs/dataset_real.yaml:77-86`).
- hm: локальные максимумы в окне 7, субпиксельное уточнение параболой по log, круг радиусом `point_radius_px` = 4 px (≈4 м при 1 м/px) из 16 вершин (`vectorize.py:206-220`, `PostConfig :38-46`).
- NMS по расстоянию центроидов внутри класса: радиус 12 px в конфиге (`configs/dataset_real.yaml:77-86`; по умолчанию в коде 6 px, `vectorize.py:45`), реализация `src/expds/post/nms.py:13-42`. Затем top-2000 (`vectorize.py:226-246`).
- Все детекции пишутся как Polygon, точки — кругами (`docs/SKELETON.md:66`). Участки объединяются простой конкатенацией (`src/expds/predict.py:71-82`).

## 4. Время, Docker, GPU
- Инференс 1024² занимает 1,6 с на RTX 3050 по синтетике (`docs/SKELETON.md:61-62`). Точка входа — `inference/entrypoint.py`, сеть отключена, `torch.load` локально (`src/expds/predict.py:1-8`).

## 5. Что у них есть, а у нас нет (сравнение с `/workspace/koz3_shared/bot-2/repo`)
| Идея | У них | У нас | Оценка выигрыша / цена |
|---|---|---|---|
| Богатый стек рельефа (openness±, SVF, TPI, TRI, curvature, SLRM σ8, плотность точек, интенсивность) | `configs/dataset_real.yaml:18-37` | сданные каналы лидара — локальный рельеф и уклон; `rough` и `curv` в конфиге не стоят (`docs/ARCHITECTURE.md:3`) | оценка: от 0 до +0,01 к итогу, в основном за счёт recall курганов на лидаре (лидар 471 TP / 912 FP, `docs/LB_LOG.md:75`). Цена высокая: переобучение `li_r18_hard2` и LOFO |
| Нормировка по участку (`site_robust`) | `:39` | не проверял, отдельно не указано | оценка: малая; проверить, что каналы между семьями не съезжают |
| Головы seg+heatmap в одной сети | `factory.py:15` | пики рельефа + U-Net в центре пика (`docs/ARCHITECTURE.md:18`), карта центров на спутнике (`:19`) | у нас по сути уже так же; оценка ≈0 |
| Круг r≈4 м на центр | `vectorize.py:206-220` | круг 4 м на местности, буфер в UTM (`docs/ARCHITECTURE.md:19`) | то же самое; ≈0 |
| Подавление по расстоянию центров | `nms.py:13-42` (12 px) | подавление в 12 м между ветками (`docs/ARCHITECTURE.md:19-20`) | то же самое; ≈0 |
| Без ImageNet | `factory.py:55` | ImageNet (`docs/ARCHITECTURE.md:46`) | у нас лучше: по нашему стеку предобучение скорее помогает. Это оценка |
| Подклассы курганов | сливаются в `mound` (`import_labels.py:53,64`) | две группы порогов и копии подклассов (`expds/post.py:472-519`) | у них хуже; учиться тут нечему |

## 6. Баги и утечки в их оценке (насколько верить их числам)
- Своя метрика не совпадает с официальной. Для полигонов она считает IoU ≥ 0,5 (`src/expds/eval/metric.py:4,42,244-245`), а официальная — покрытие ≥ 0,5 от меньшей фигуры плюс representative point внутри GT. Для точек допуск `r_tol` = 15 в единицах CRS, с пометкой «уточнить по регламенту» (`configs/dataset_real.yaml:91-92`, `metric.py:43-44,199-206`), а официальный допуск 2 м. Метрика безклассовая/по классам без весов, `__all__` (`train.py:125`, `metric.py:35`). Их F1 с LB сравнивать нельзя.
- Числа на синтетике (`docs/SKELETON.md:59-62`) к реальным данным отношения не имеют.
- Ключи `class`/`score` и классы `mound/rampart/ditch` в выводе обнуляют результат на официальной метрике (§0).
