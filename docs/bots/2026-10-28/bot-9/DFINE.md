# D-FINE-L на SpOr (детектор рамок): ядро Kaggle (бот 9)

Статус на 28.10.2026 21:38 MSK: ядра **собраны, НЕ запушены** (пушит бот 1). В сдачу результат пойдёт только после письма организаторов о весах.

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
- Веса HF `ustc-community/dfine-large-coco`, ревизия закреплена: `aa8f8211545e4cff5adf52ea512f9ac74a9824ec`. Лицензия Apache-2.0, предобучение на COCO, модель D-FINE (Peng et al., 2024), 31,36 млн параметров.
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

## Точка подключения тренера бота 5
- Если рядом со скриптом или в `/kaggle/input/**` лежит `dfine_train_bot5.py` с функцией `train(model, batches, cfg, device, log) -> model`, используется она. Иначе работает встроенный тренер по рецепту §6.1.
- На 21:38 MSK в `/workspace/koz3_shared/bot-5/` такого скрипта нет. Жду от бота 5 либо этот файл (его надо положить в датасет и добавить в `dataset_sources`), либо подтверждения, что встроенный тренер годится.

## Проверки (box, CPU, ≤2 потока, nice 10)
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

## Оценка времени
- Скорости на T4 ещё нет. Цифра с CPU box (6,66 с/ит на 256 px, bs 1) для GPU не годится. Smoke (~10 ит + 1–2 маленьких участка) должен занять порядка 15–25 мин с учётом pip и скачивания весов (≈125 МБ fp32). Это оценка, не замер. Smoke печатает `ПРОГНОЗ full: с/ит → 8000 ит × 5 фолдов / 2 GPU`.
- Full ограничен дедлайном 11,3 ч. 5 фолдов делятся на 2 GPU (3 «волны»). На 50-й итерации тренер пересчитывает число итераций под бюджет фолда, на инференс фолда резервирует ~20 мин и останавливается за 30 мин до дедлайна. Если на T4 с/ит > ~0,8, будет меньше 8000 итераций — это будет видно в `dfine_report.json`.

## Риски
1. NaN в fp16 / DN-ветке → `DFINE_NUM_DN=0` (HF PR #45601 чинит aux-лоссы при DN=0; в transformers 5.21 не проверял).
2. Принудительная установка transformers==5.21.0 может конфликтовать с huggingface_hub/tokenizers образа Kaggle. Ранний фейл на импорте поймает это в первые минуты.
3. Окна только с фоном: при num_boxes=1 VFL-loss большой (~580). Смягчено долей объектных окон 60 % и clip 0.1.
4. Инференс полных растров: резерв ~20 мин на фолд может не хватить на больших участках. Тогда урезается обучение следующих фолдов; есть общий дедлайн и лог прогресса каждые 200 окон.
5. Городища крупнее окна 320 м не поместятся в рамку. Для них детектор слаб: ожидаемо, лидара нет.
6. Чипы JPEG в `/tmp/chips`: ≤400 объектных + фон на участок, на диске порядка единиц ГБ. Не замерено на полном наборе.
7. Двух GPU-ядер одновременно запускать нельзя: второе молча получит CPU и упадёт по `--require-gpu`.
8. Веса — только для замера: в сдачу после письма организаторов (п. «Веса и лицензия»).
