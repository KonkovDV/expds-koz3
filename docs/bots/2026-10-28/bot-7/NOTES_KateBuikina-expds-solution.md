# NOTES — KateBuikina-expds-solution (bot-7, только чтение)

Репозиторий: `/workspace/koz3_shared/competitors/KateBuikina-expds-solution`, HEAD 7a6c7bb «add readme» (2026-05-27 20:29 MSK). Ничего не запускалось.
Пути ниже — относительно корня репозитория; `fs/` = `final_solution/`.

## 0. Главное
- **Это курсовая работа (README.md:3-4) по данным КОЗ №2**: данные `../train-expds-2` (README «Данные»; `solution-li/config.yaml:7`), метрика в `solution_orto/utils/metric.py:1-11` — «IoU > порога ИЛИ центроид внутри GT», веса КОЗ2 0,278/0,222/0,167/0,111/0,056 (`:24-30`), порог IoU 0,3 (`:104`). Не КОЗ3.
- **Ни одного заявленного балла в репозитории нет** (нет логов/CSV/таблиц результатов; `find` по *.csv/*.md/*.json/*.log — только README, ai_declaration, calibration.json, metadata.json). Подтвердить 0,5–0,6 из кода нельзя.
- **Критический баг выдачи (по коду):** имя класса в выдаче получает суффикс модальности — `_qualified_class_name` (`fs/pipeline_core/segmenter.py:49-54`) → `kurgany_tselye_Li`, `object_poly_Mag`, `…_Sat/_Las/_Orto` (вызовы `:562, :819, :1628, :1639`). Экспорт пропускает такие имена как допустимые (`fs/utils/geo_export.py:29-33, :339`) и пишет как есть (`:360-364`). Официальные метрики (и наша `materials/compute_metrics.py:158-165, :827-830`, и КОЗ2 recara `compute_metrics.py:105-112, :347-350`) такие имена не нормализуют и **отбрасывают как не целевые** → под любой из метрик их выдача давала бы ≈0. Значит, если у этой команды был балл 0,5–0,6, он получен не этой версией кода (гипотеза).

## 1. Модели, вход, предобработка (`fs/configs/*.yaml`, README «Что сделано»)
| маршрут | вход | модель | тайл | нормализация |
|---|---|---|---|---|
| MAG (`maggr`) | 1 канал | свой U-Net, base 32, depth 3, dropout 0,2 (`fs/configs/mag.yaml`) | 384, перекрытие 64 | перцентили 1–99 + z-score **на патч** (`fs/pipeline_core/mag_preproc.py:19-30`) |
| LAS/LAZ | 4 канала из облака: z_top, …, z_span, шаг 0,5 м (`fs/utils/las_grid.py:101-122`, `fs/configs/las.yaml:3-4`) | U-Net ResNet34 | 512 | min-max на тайл (`las_grid.py:106`) |
| LI | 4 tif `_c/_ch/_g/_i` (готовые отмывки) | U-Net ResNet34, ImageNet (`fs/configs/li.yaml:3,17-18`) | 512 | min-max **на патч по каналу** (`segmenter.py:1070-1074`; обучение — то же, `solution-li/dataset.py:649-652`) |
| SAT (SpOr, ae) | 4 канала | U-Net ResNet34 | 384 (`fs/configs/sat.yaml:3`) | — |
| ORTO (Or) | RGB | U-Net ResNet34 | 640, перекрытие 0,07 (`fs/configs/orto.yaml`) | температура 0,85 (`solution_orto/models/calibration.json`) |
- Классы: 5 «рельефных» + фон для LI/LAS/SAT/ORTO; MAG — фон/object_poly/finds_points (`fs/configs/mag.yaml`). Маршрут — по имени файла + классификатор модальности (`fs/pipeline_core/router.py`, `fs/configs/pipeline.yaml`: веса `classifier_mobilenet_v3_small.pt`, но `classifier_arch: shufflenet_v2_x0_5` — несоответствие имён, мелочь).

## 2. Обучение
- LI (`solution-li/config.yaml`): ce_dice (0,8/0,6), веса классов [0,15, 8, 12, 3, 8, 6] (`:47, :55-59`), AdamW 1,2e-4, cosine+warmup, 80 эпох, batch 2, early stop 14 по `val_iou_fg` (`:61-81, :106`); аугм. flip h/v, rot90, яркость/контраст 0,1, шум (`:83-96`).
  - Выбор кропа: до 12 случайных попыток, берётся кроп с max числом пикселей объекта (`solution-li/dataset.py:562-596`) — сильное пересэмплирование объектов.
  - Валидация: одно окно 512 с **максимумом переднего плана** на растр (`dataset.py:666-728`) → пиксельный IoU оптимистичен.
  - Сплит: по растрам LI (пара = группа `_c/_ch/_g/_i`), стратифицировано по классам из имён geojson (`solution-li/dataloader.py:218-277`), **не по регионам** → соседние растры одного региона в train и val (утечка, гипотеза о масштабе).
- SAT: focal_dice, веса [0,03, 8, 18, 6, 10, 6] (`solution-sat-ae/config.yaml:51, :65-66`). LAS: focal_dice, веса [0,002, 100, 20, 15, 300, 80] (`solution-las/config.yaml:29, :37-38`).
- ORTO (`solution_orto/configs/pipeline.yaml`): focal γ=2 + dice, label smoothing 0,08, mosaic 0,42, recall_fg_prob 0,85, веса фокала по классам [1, 1, 1,35, 1,35, 1,45], lr 6e-5, wd 0,08; **сплит по регионам** (`solution_orto/run_train_expds.py:150-178, :341`) — единственный честный сплит в репо.
- Заявленных метрик и сплитов с числами — нет.

## 3. Постобработка (`fs/pipeline_core/segmenter.py`, `fs/utils/vectorizer.py`)
- Логиты усредняются по перекрывающимся окнам → softmax → argmax по fg-классам + пороги: общий и по классу, свои для каждой модальности (`segmenter.py:335-355`; `fs/configs/pipeline.yaml:16-25`: LI [0,32, 0,34, 0,45, 0,54, 0,54], SAT [0,34…0,50], LAS [0,38…0,54]).
- Вектор: `cv2.findContours(RETR_EXTERNAL)` → полигон по контуру маски (`vectorizer.py:49-77`); confidence = **max** вероятности в bbox контура (`:66-76`). Мин. площадь: 18 м² (общая), LI 40, SAT 40, LAS 55 (`pipeline.yaml:20-31`); опционально морф. открытие (`segmenter.py:60-98`).
- MAG: чистый argmax, **без порогов и без confidence** (conf = 1,0; `segmenter.py:879-881`), и общий `min_area_m2 = 18` (`segmenter.py:1671-1673`) → **все finds_points (эталон ≤2 м) отрезаются** порогом площади.
- Крупные растры LI векторизуются тайлами 4096 с перекрытием 512 (`segmenter.py:478-526`) **без дедупликации** → дубли на стыках.
- Слияния/дедупа между модальностями нет; `markup_type` в выдачу не пишется (`geo_export.py:360-364`); лимиты 20000/3000 не обрабатываются.
- Экспорт в EPSG:3857 с эвристиками «похоже на WebMercator/UTM — пишем как есть» (`geo_export.py:168-286`) — риск неверных координат.

## 4. Время, Docker, GPU
- `fs/Dockerfile`: `pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime`, все *_NUM_THREADS=1, `INFERENCE_TIME_BUDGET_SEC=1500`; образ `sgromovtech/archeology-inference:v1` (`fs/metadata.json`). Нужен CUDA GPU (README). Частичная запись при исчерпании бюджета (`fs/inference/main.py` ~:303-400). Размер образа неизвестен.

## 5. Что у них есть, чего нет у нас (сравнение с /workspace/expds-koz3)
- Почти всё уже есть у нас: U-Net smp (`expds/train.py:497`), отмывки c/ch/g/i (`expds/lidar.py:42, :413-435`), пороги по классу и markup_type, бюджет времени.
- Отличия, которые можно дёшево проверить (все — гипотезы, прироста не измеряли):
  1. **Энкодер ResNet34 с ImageNet** вместо нашего ResNet18 по умолчанию (`expds/train.py:412, :707`) на LI — цена: переобучение на GPU.
  2. **Кроп с максимумом объектов** при обучении (`solution-li/dataset.py:562-596`) — у нас схему сэмплирования не сверял.
  3. **Температурная калибровка** (ORTO, 0,85) — для нас бесполезна: пороги и так подбираются по классу/группе, монотонное преобразование ранги не меняет.
  4. LAS-растеризация «z_top/z_span/intensity» вместо отмывок — альтернативный вход; у нас DEM строится из LAS сами.
- Не переносить: полигон = контур маски (по REPORT.md §2 копия формы хуже маленького диска в якоре), confidence = max в bbox.

## 6. Баги / утечки
- **Суффикс модальности в class_name** → выдача не засчитывается метрикой (см. §0).
- finds_points не выживают (min_area 18 м² для MAG).
- MAG: confidence = 1,0 у всех объектов → кривая PR вырождена.
- Дубли на стыках тайлов векторизации LI.
- Сплит LI/SAT/LAS по растрам, не регионам; валидация по окну с max объектов — оптимистичный pixel IoU; метрика в `solution_orto/utils/metric.py` — КОЗ2.
