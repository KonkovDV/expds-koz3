# AEOR_DETECT — все-классовый D-FINE на Ae/Or в линии bot-7 (bot-6, 29.10)

**Ветка:** `bot6-aeor-detect` от `360c870` (bot7-candidate-v3, код = 36af6b7), коммит **`3617be6dceb38cb4b7045d20ca81d91e38c8f209`** (только локально).
**Diff:** `bot-6/bot6-aeor-detect.diff` (1154 строки; 6 файлов, +1016/−11 по `git diff --stat 360c870..bot6-aeor-detect`).

## 1. Ревизия скелета (aeor_dfine.py → expds/dfine_aeor.py)
- В скелете был `S.DFineUnavailable`, которого в v3 нет. Заменён на собственный `AeOrUnavailable`, маркер `DFINE_AEOR_UNAVAILABLE`.
- Разрешение Or 0,25 м исправлено на 0,5 м, как в рецепте bot-9: DFINE_FINAL_ORAE.md, dfine_main.py:53,66.
- Tile и overlap были глобальными. Теперь они заданы отдельно для каждой модальности: Ae 640/160, Or 1024/288.
- Прямой импорт transformers убран. Модели грузятся только через `sp_dfine.load_models` → `import_dfine` (проверка num_labels==5).
- Имена: флаг `[dfine_aeor]`, markup_type нативный `ae`/`or`, threshold_group `dfine_ae`/`dfine_or`, пороги `[thresholds_by_markup.dfine_ae/dfine_or]`.

## 2. Точка вставки (expds/pipeline.py)
- Фаза `_dfine_aeor_phase` (стр. 1129) вызывается после `_sp_dfine_phase` (стр. 1337) и до gor_gate (стр. 1344). Импорт ленивый, `phases.append("dfine_aeor")`, `state.add`.
- На каждый регион вызывается `_soft_ok(guard,"dfine_aeor",...)`. Логи: `dfine_aeor n_det=N region=…` и `dfine_aeor n_det=N total`.
- Preflight в run() работает только при `enabled`:
  - проверяет импорт через sp_dfine.import_dfine, наличие файлов весов, соответствие id2label классам compute_metrics и готовность порогов;
  - при проблеме пишет ERROR с маркером (check_submission → FAIL), и сабмит собирается без фазы.
- Кэш моделей общий с sp_dfine (`dfine_cache`, стр. 1332). Если пути совпадают, модели не грузятся повторно.

## 3. Дизайн
- **Тайлинг.** Тайлы нарезаются в разрешении обучения через Scene/WarpedVRT в UTM. Последний ряд и столбец прижимаются к краю. Пустые тайлы (доля nodata ≥ 0,95) пропускаются.
- **Пороги.**
  - Детектор работает с низким полом `min_score` 0,05.
  - Пороги по группам и классам берутся из `thresholds_file` (JSON `{"dfine_ae":{класс:порог},"dfine_or":{...}}`) и сливаются в `thresholds_by_markup`; при конфликте выигрывают ключи из TOML.
  - При `require_thresholds=true` фаза без порогов не запускается: так нельзя выпустить детекции с полом 0,05.
  - Сейчас порогов нет. Это честная заглушка для bot-10, цифры не придуманы.
- **TTA и масштабы.** TTA — горизонтальный flip. `scales` по умолчанию `[1.0]`, опционально `[1.0, 1.5]`.
- **Слияние детекций.** WBF (IoU 0,55) или NMS (0,5). Сливаются только виды одного растра одной модальности, между снимками и типами слияния нет (это проверено тестом).
- **Геопривязка.** Боксы переводятся из UTM в 3857 и становятся эллипсами (32-угольник), затем `cluster_circles` внутри растра. Классы сверены с `materials/compute_metrics.py`.
- **Time guard.**
  - `plan_level` проходит уровни full → no_flip → one_scale → skip с запасом ×8 на измеренное `t_view` и выбранный уровень пишет в лог.
  - Бюджет фазы — 120 с на датасет (`budget_s`), регионы идут от дешёвых к дорогим.
  - `max_objects` 20000 на группу.
- **Упаковка.** build_submission кладёт модуль, sp_dfine, vendor, веса и thresholds_file только при `enabled`. check_submission ищет маркер, а при включённом флаге требует наличие модуля и vendor и проверяет импорт.

## 4. Проверки
- `tests/test_dfine_aeor.py`: **29 passed** (`bot-6/work/aeor_pytest_final.log`), детектор заглушечный. Что покрыто:
  - тайлинг (покрытие и перекрытие);
  - WBF/NMS не сливает разные снимки и типы;
  - round-trip геопривязки;
  - деградация guard и её логи;
  - дедлайн фазы;
  - merge_within_markup;
  - вариант R не удаляет dfine_or;
  - frame_prior не трогает dfine_*;
  - verify не сбрасывает conf;
  - пороги групп;
  - гейтинг, ленивый импорт и порядок фаз;
  - без флага ничего не упаковывается;
  - маркер в check_submission.
- **Побайтовая идентичность** (`bot-6/work/aeor_identity/check_identity.py`, результат в `run2/identity.json`). На синтетике conftest (LAS + MagGr) сравнивались sha256 выхода `pipeline.run` в трёх режимах:
  - база 360c870;
  - ветка с выключенным флагом;
  - ветка с флагом без весов.

  Все три sha = `9222f5d1…9a83d9`, n_features=245, маркер залогирован. Первый прогон run1 упал на базе из-за лимита 2 ГБ AS (`memory allocation of 60000000 bytes failed`). Повтор прошёл с MALLOC_ARENA_MAX=2 и RAYON_NUM_THREADS=2.
- **Смежные тесты при prlimit 2 ГБ.** Все падения — из-за окружения: `can't allocate memory`, `failed to map segment` (scipy, lazrs).
  - v3_flags, dfine_import, check_submission, crop_verify, candidate_v2: на ветке 102 passed / 9 failed (`work/aeor_pytest_flags.log`). Все 9 упавших — в crop_verify и candidate_v2, где torch и scipy упираются в лимит памяти. На базе pytest упал с INTERNALERROR (`work/aeor_pytest_flags_base.log`).
  - test_pipeline и др.: 5 failed / 43 passed на ветке и те же 5 на базе (`work/aeor_pytest_pipeline.log`, `work/aeor_pytest_pipeline_base.log`).

## 5. Как включить
```toml
[dfine_aeor]
enabled = true
weights = { ae = ["models/dfine_ae"], or = ["models/dfine_or"] }   # HF-папки: config.json + model.safetensors
thresholds_file = "models/dfine_aeor/thresholds_oof.json"          # {"dfine_ae": {класс: порог}, "dfine_or": {...}}
# scales = [1.0, 1.5]
# budget_s = 120
```

## 6. Что ждём от bot-9 / bot-10
- **bot-9:** чекпойнты Ae и Or (HF-папки, 5 классов sp_dfine.CLASSES) и замер времени на тайл на 4090.
- **bot-10:** вложенные OOF-пороги в формате выше. Пороги arch для Or слабые, потому что fold 0 не видит arkhitektury.

## 7. Вопросы к bot-7
1. `merge_within_markup` объединяет круги dfine_ae с другими ae-источниками того же класса (union, max conf). Так и задумано, или нужен отдельный ключ markup/source?
2. Будет ли ребейз на v4? Сейчас ветка идёт от 360c870.
3. Нужен ли потоковый detect_tif для больших Or-растров? Сейчас растр читается тайлами через Scene, память ограничена батчем.
4. Реальный tile_s на 4090 нужен, чтобы откалибровать ×8. Стоит ли включать scales 1,5?
5. Для dfine_aeor в check_submission нет smoke-эталона (аналога DFINE_KNOWN). Добавлять ли его?
