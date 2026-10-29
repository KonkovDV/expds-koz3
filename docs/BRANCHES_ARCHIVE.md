# Архивные ветки (29.10.2026)

Ветки **не влиты** в `main` и **не удалены**. На их головах поставлены аннотированные теги `archive/*`
(запушены), чтобы содержимое не потерялось, если ветки потом удалят. Удаление веток — только решение владельца.

| Ветка | Голова | Тег | Впереди `main` 164707a |
|---|---|---|---|
| `bots/bot8-cpu-a-clean` | `95bc0a6a357d039872fa03769fa3265240c93e89` | `archive/bots-bot8-cpu-a-clean` | 8 коммитов |
| `bots/bot8-cpu-a-clean-on-v2` | `18b2f66acafd1d65329460376a58af0dd810bcef` | `archive/bots-bot8-cpu-a-clean-on-v2` | 9 коммитов |
| `speedup-2026-10-28` | `6602bb23a88076969e909e674f9a896070db1c38` | `archive/speedup-2026-10-28` | 1 коммит |

Почему не влиты:

* `bots/bot8-cpu-a-clean`, `bots/bot8-cpu-a-clean-on-v2` — **не clean room**: `docs/CLEANROOM_SPEC_lidar_a.md`
  написан тем же агентом по поведению EhimenNathan-EXpedition @d434e56; `tests/lidar_a_oracle.py` /
  `tests/test_lidar_a_oracle.py` загружают репозиторий конкурента. Плюс бинарь
  `rust/expds_fast/wheels/expds_fast-0.1.0-cp312-abi3-manylinux…whl`.
* `speedup-2026-10-28` — меняет пути CPU по умолчанию («байт-в-байт с 14d419…» заявлено, но не проверено
  на v2/main) и добавляет тот же `.whl`.

Восстановить: `git checkout -b <имя> archive/<тег>`.

## Харвест 29.10.2026 (ветка `harvest/2026-10-29`, слияние `--no-ff` в `main` f1b8d4c)

Вопрос владельца «есть что-то ценное?». Разобран каждый дифф против `main`. Правила: ничего из линии EN
(спека/тесты ссылаются на поведение или репозиторий конкурента — не берём), никаких `.whl`/бинарей/данных/весов,
умолчания `main` = v2 (новое — по умолчанию ВЫКЛ.).

| Ветка → часть | Решение | Почему |
|---|---|---|
| `speedup` → `infer_nn` meta-init (сеть на meta + assign весов) | **ВЗЯТО**, флаг `meta_init` | чисто, тот же state_dict и выход (`tests/test_meta_init.py`); + откат на обычный путь, если веса не float32 |
| `speedup` → `infer_nn` без копии `np.pad`, когда паддинг 0 | **ВЗЯТО**, флаг `no_pad_copy` | чисто, те же данные |
| `speedup` → `infer_nn._predict_on_device_slots` (+`_axis_slots`, `_slots_ok`, `slot_bytes`) | **ВЗЯТО**, флаг `slot_accum` (только явно, не входит в `enabled`) | побитно = `_predict_on_device` (`tests/test_predict_slots.py`); на CPU медленнее, на CUDA не замерено |
| `speedup` → `pipeline._read_ahead` (чтение следующего окна в фоне) | **ВЗЯТО**, флаг `read_ahead` | чисто; перенесено поверх дедлайна в цикле окон и `zero_nodata` из `main` (выкл. — чтение после проверки дедлайна, как было) |
| `speedup` → `raster.circles_3857` / `_transform_many`; `centers`, `geophys` | **ВЗЯТО**, флаг `batch_circles` | побитно те же WKB (`tests/test_circles_batch.py`); `lru_cache` — только у копии для пакетного пути, `circle_3857` не тронут |
| `speedup` → `writer._polygon_coords` | **ВЗЯТО**, флаг `fast_writer` | тот же JSON до байта; сохранены пропуск NaN и `fid = len(out)` из `main`. Тест переписан: сравнение флаг вкл./выкл., без `git show 14d419a` (не работал на shallow CI) |
| `speedup` → `tiles.fix_channels` без repeat+concatenate | **ВЗЯТО**, флаг `fast_channels` | те же массивы |
| `speedup` → `GDAL_NUM_THREADS=ALL_CPUS` в `inference/main.py` и `Dockerfile` | **ВЗЯТО иначе**: флаг `gdal_threads` (ставит env, если не задан) | `inference/main.py` и `Dockerfile` не тронуты — умолчание сдачи прежнее |
| `speedup` → `expds/las_grid_fast.py`, `rust/expds_fast/*`, `tests/test_las_grid_fast.py`, `fast_so()` в `build_submission.py`, исключения `.gitignore` для колеса | **НЕ ВЗЯТО** (EN) | написаны как «замена ground.rasterise для DTM lidar_a» — линия EN; в `main` потребителя нет |
| `speedup` → `expds/cpu_threads.py` (+тест) | **НЕ ВЗЯТО** | потребитель только `lidar_a` (EN) |
| `speedup`, `cpu-a-clean*` → `expds_fast-0.1.0-cp312-abi3-manylinux….whl` | **НЕ ВЗЯТО** (бинарь) | бинарь в git запрещён |
| `cpu-a-clean*` → `lidar_a/*`, `docs/CLEANROOM_SPEC_lidar_a.md`, `tests/lidar_a_oracle.py`, `tests/test_lidar_a*.py`, `build_submission_git` (колесо) | **НЕ ВЗЯТО** (EN) | спека по поведению EhimenNathan-EXpedition, оракул грузит его репозиторий |
| `cpu-a-clean*` → коммит ускорений | **НЕ ВЗЯТО** (дубль) | подмножество `speedup` (las_grid_fast, cpu_threads, expds_fast, колесо) |
| `cpu-a-clean*` → `check_submission.check_so` glob `expds_core*.so` → `expds_*.so` | **НЕ ВЗЯТО** | нужно только для `expds_fast`, которого в `main` нет |
| `bots/bot8-li-half` (`e449736`) → `expds/li_half.py`, `[li_half]`, проход в `pipeline`, `cross_class` в `drop_near_kurgans`/`dedupe_cnx`, `sigma_anchors`, проверки `check_submission`, тесты | **НЕ ВЛИТО**, остаётся на ветке | эксперимент отклонён honest-10 (HX +0,0004, P 0,67); `dedupe_cnx` меняет поведение по умолчанию |

Ускорения включаются `[speed] enabled = true` в конфиге или `EXPDS_SPEEDUP=1` (по одному — `EXPDS_SPEED_<ИМЯ>=1`),
описание — `expds/speed.py`. Синтетический смоук (4 участка: LAS, MagGr, два Ae-снимка, `inference/main.py`):
выход **байт-в-байт** одинаковый — `main` f1b8d4c, харвест выкл., харвест вкл., вкл. + `slot_accum`
(sha256 `90b92d3f…`). На реальных данных и 4090 не перемерено — поэтому по умолчанию выкл.
Ветки `speedup-2026-10-28` и `bots/bot8-li-half` не удалены.
