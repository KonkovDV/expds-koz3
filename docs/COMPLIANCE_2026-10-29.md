# Сверка main с утверждённым ТР — 29.10.2026

База: `main` @ `f1b8d4c`, ветка `compliance/tr-approved-2026-10-29`. Источники: утв. ТР (`docs/background/TR_KOZ3_approved.txt`),
консультация №2 (`docs/background/KOZ3_consult2_answers.txt`), отличия от проекта — `docs/background/TR_DIFF_draft_vs_approved.md`.
Выжимка правил — `docs/RULES.md`. Номера строк — по ветке на момент сверки.

## Критичные риски (не закрыты кодом)

1. **Данные КОЗ на Kaggle** (13.10.1–13.10.2, консультация №2: Kaggle/Colab «не автоматически разрешены»). Ядра создаются с
   `is_private: true`, но датасеты с выборкой/эталоном надо проверить вручную (Private, без соавторов вне команды) и удалить после работ.
   На боксе в `/workspace/kaggle_out` лежат выгрузки ядер (OOF, gt_*). Действие — за командой.
2. **Сборка архива из чистого клона main падает:** `configs/default.toml:430` `[rerank_arch] enabled = true`, а `models/rerank_arch_lgbm.txt`
   по правилу «веса не в git» исключён из main (коммит `3607b2e`). `scripts/build_submission.py` завершает работу с
   «модель из конфига не найдена». Файл лежит вне git (md5 `9a9f7f75…`, например `koz3_shared/bot-8/wt-*/models/`). Перед сборкой положить его
   в `models/` или выключить флаг. Код не менялся: выбор между вариантами меняет выход сабмита.
3. **Лицензионные решения со слов, без письменной фиксации:** ImageNet/COCO/DINOv2/Objects365/ADAF — «разрешено 29.10 по сообщению
   бота 1». Для 12.3/13.5 нужна ссылка на письменный ответ организаторов. `expds/geo_pixgbm.py` (сходство с публичным решением
   конкурента) в main не влит — `expds/pipeline.py:359–363` при его отсутствии уходит в классику.
4. **Точные версии CUDA/cuDNN образа** `pantum/n7q:20261024` не сняты (консультация №2 требует компоненты и версии) — TODO в `docs/THIRD_PARTY.md`.

## Чек-лист

| # | Требование | Итог | Доказательство | Что сделано |
|---|---|---|---|---|
| 1 | Лидар на валидации только LAS (Прил. 1 п. 4); Li не используются; LAS→DEM→сеть; без LAS — тихий пропуск | **OK** | `expds/catalog.py:33` (`li_rasters` — «только трейн»), `:152`; `li_rasters` читается только в `expds/tiles.py:57` (обучение); `expds/pipeline.py:469` `for las in reg.las` (ЦМР из LAS), `:866` без LAS ветка CNX не запускается | тесты `test_region_with_las_only_has_li_type`, `test_li_tif_never_reaches_inference_rasters`, `test_lidar_branch_runs_from_las_without_li` |
| 2 | Дедупликация внутри типа данных, между типами не склеивать | **OK** | `expds/post.py:99–111`: группа (регион, `markup_type`, класс) | тест `test_merge_within_markup_does_not_cross_types_and_keeps_etgr` |
| 3 | EtGr: каждая карта — отдельные объекты | **OK** | `expds/post.py:111` (`etgr` не сливается); README «Геофизика» | тот же тест |
| 4 | ≤ 100 МБ, ≤ 20 000 объектов, ≤ 3000 вершин | **было частично → исправлено** | `expds/constants.py:67–69`; `expds/writer.py:56` (`_rounded_coords`), `:118` (`SIZE_BUDGET_BYTES` 97 МБ), `:125` (`_fit_size`), `:163–275` (замечания Прил. 3, > 50 → ошибка); `scripts/check_submission.py:1222,1288` | Защиты размера не было, валидность после округления до 2 знаков не проверялась, замечания Прил. 3 не считались. Добавлено, тесты `test_rounding_*`, `test_invalid_geometry_dropped_not_written`, `test_tr_remarks_*`, `test_size_guard_*` |
| 5 | Контейнер: `metadata.json`, 30 мин, архив ≤ 2 ГБ, `/dev/shm` 16 ГБ, PID ≤ 8192 с потоками, EPSG:3857 | **OK** | `metadata.json` (`image`, `entry_point`); `expds/constants.py:73`, `expds/watchdog.py:344`; `expds/writer.py:42` EPSG:3857; `inference/main.py:21` OMP 8; пулы потоков ограничены (`expds/fastgeom.py:101`, `expds/lidar.py:352`); `scripts/check_submission.py` проверяет размер и структуру | без изменений. Папка `inference/` в утв. ТР больше не обязательна — наша структура валидна |
| 6 | Метрика Прил. 4 = `compute_metrics.py` | **OK** | `materials/compute_metrics.py` совпадает с файлом организаторов (md5); веса классов 30/20/15/10/10/10/5 — README, `expds/constants.py` | без изменений |
| 7 | Лицензии и веса: 12.3, 12.5, 13.4, 13.5, 13.10 (вкл. 13.10.3), NVIDIA, AGPL | **частично → дописано** | `docs/THIRD_PARTY.md` (ADAF CC BY-SA, разделы NVIDIA и Kaggle/Colab); детерминизм 13.4: `expds/infer_nn.py:21,28–37`, `configs/default.toml:35`; ultralytics (AGPL) не используется | Добавлены компоненты NVIDIA (драйвер с хоста, CUDA/cuDNN в колёсах torch cu128), правила облака по консультации №2, статус ADAF. Открыто: версии из образа, риски 1 и 3 |
| 8 | Прил. 6 ПКК → план; README/декларация по шаблонам | **OK (план)** | `docs/PKK_GAP.md` (≈ 45 чел.-дней), `docs/README.md` и `docs/ai_declaration.md` по шаблонам платформы | Созданы. Контакты (email) и версии Claude Code / агента — «заполнить» человеком |

Ещё: README утверждал, что неизвестный класс отклоняет файл. По утв. ТР (Прил. 1) такой объект пропускается — README исправлен.

## Аудит-триаж диффа

| Изменение | Зачем | Риск | Чем проверено |
|---|---|---|---|
| `expds/writer.py` — `_rounded_coords`: пробует 2→4→6 знаков, берёт первое валидное с площадью > 0, иначе объект отбрасывается | После округления до 2 знаков тонкий полигон мог стать самопересекающимся, а это замечание Прил. 3 | Низкий: на нормальных полигонах результат прежний (2 знака) | тесты; **побайтно совпадает со старым writer** на 3 реальных выходах (9015, 9015 и 47 518 объектов, sha256 совпадает) |
| `expds/writer.py` — `_fit_size` (бюджет 97 МБ, cap × 0,9) | Файл > 100 МБ отклоняется целиком; защиты не было | Низкий: включается только при > 97 МБ, режет по весам классов (`cap_features`) | `test_size_guard_noop_for_normal_output`, `test_size_guard_trims_by_class_quota` |
| `expds/writer.py` — `tr_geometry_remarks`, `validate()` > 50 → ошибка | Правило 50 замечаний (Прил. 3) | Нет: только проверка | `test_tr_remarks_detected_and_limit`, `test_written_file_has_no_remarks`; на реальных выходах 0 замечаний |
| `expds/writer.py` — проверка `isfinite` | NaN/Inf — замечание Прил. 3 | Нет | тесты |
| `scripts/check_submission.py` — `tr_remarks` | Проверка архива до отправки | Нет: > 50 → FAIL, иначе WARN | полный pytest |
| Документы (README, RULES, COMPLIANCE, PKK_GAP, THIRD_PARTY, docs/README, docs/ai_declaration, AI_DECLARATION, TR_DIFF, CHANGELOG, утв. ТР и консультация в `docs/background/`) | Требования Прил. 5 и сверка | Нет | вычитка |

Новых фич нет, флаги и пороги не менялись. Выход сабмита на реальных данных не изменился (побайтная сверка выше).

## Тесты

`OMP_NUM_THREADS=2 nice -n 10 pytest -q` — на ветке до мержа **582 passed, 33 skipped**; после мержа с main `58ab645` (harvest, флаги [speed]) — **625 passed, 33 skipped** (12 тестов в `tests/test_tr_compliance_2026_10_29.py`). При конфликте с `fast_writer` (флаг `EXPDS_SPEED_FAST_WRITER`, по умолчанию OFF) быстрый путь тоже проходит проверку `_ring_ok`: координаты конечные, полигон валиден, площадь > 0; иначе объект идёт медленным путём с округлением 2→4→6 знаков. При включённом флаге выход совпадает с выключенным (тест `test_fast_writer_path_also_keeps_thin_polygon_valid`).
Для `test_coco_roundtrip` в окружении бокса доставлен `opencv-python-headless`, код не менялся.
Сборка и проверка архива `build_submission.py` + `check_submission.py --no-smoke` на чистом клоне не прошли из-за риска 2.

## Не сделано (сознательно)

- Повышать число потоков и задействовать «все CPU/GPU» не стал. Это меняет время и, через порядок вычислений, может сдвинуть выход
  (Делоне и детерминизм cuDNN). Нужен замер на 4090/24 ядрах платформы. Запас по PID огромный: OMP 8 при лимите 8192, так что это вопрос скорости, а не правил.
- `cap_features` режет по весам классов, но не учитывает `markup_type`. На текущих выходах лимит в 20 000 срабатывает только на синтетике и стеках ботов.
- `scripts/build_submission.py` (архив для инференса на Платформе) README и ai_declaration не кладёт, и я это не менял: выход и состав архива тестов остаются прежними. Для **Финальной версии Исходного кода** (Прил. 5 п. 1.1–1.2, стр. 2228–2234) `docs/README.md` и `docs/ai_declaration.md` должны лежать в архиве — добавить при сборке финальной версии.
