# bot-8 — ускорение метрики КОЗ №3 (Rust + векторизация NumPy)

Обновлено: 2026-10-28 19:56 MSK. База «до»: `/workspace/expds-koz3` @ 24bd916 (main, только чтение).
«После»: моя копия `bot-8/repo` (незакоммиченные правки поверх ветки `bot8-reliability`; их коммиты
метрику не трогают). Патч: `patch/bot8-rust-metric.diff` (sha256 d66e8828…), чисто ложится на main 24bd916
(проверено `git apply` на свежем клоне). Все замеры: `nice -n 10`, 2 потока (env.sh), машина нагружена
другими задачами → повторы и медианы; разброс между сериями до ±30 %.

## Вывод в 4 строки
1. Время `scoreboard.py --honest` на ~80 % уходит в `build_curves` → `_collect` (построение shapely-полигонов
   из GeoJSON по одному, PROJ, GC). Сама жадная сверка (STRtree), подбор порогов — единицы процентов.
2. Построение геометрии перенесено в Rust (`flatten_polygons`) + `shapely.from_ragged_array`: **×4,7–4,8** на этом
   шаге; весь `build_curves` **×2,2** (3,76 → 1,72 с, медиана 6 повторов). Отключён GC внутри `_collect`.
3. Честный балл и бутстрепы переписаны на NumPy (целочисленные суммы → побайтно те же числа):
   на полном объёме (1,39 млн предсказаний) `honest_counts` **×9,9**, `bootstrap` **×15**,
   `paired_bootstrap` **×63**, `paired_bootstrap_counts` **×20**.
4. Тесты равенства: 50 новых (побайтно: WKB геометрий, conf/hit, JSON-результаты с порядком ключей) — все PASS;
   весь набор тестов репозитория 311 PASS, 1 FAIL не мой (нет `cv2` в venv).

## 1. Профиль (где время)

### 1.1 scoreboard --honest, шаги (выборка: каждое 20-е предсказание стека = 69 336 шт., весь эталон 10 979 шт.)
`tools/harness.py` повторяет шаги scoreboard после сборки стека. Исходный код, `prof/harness_base.json`:

| Шаг | с | % |
|---|---|---|
| build_curves (метрика: геометрия + UTM + сверка) | 4,905 | 80,1 |
| honest_counts (10 фолдов) | 0,464 | 7,6 |
| paired_bootstrap (1000) | 0,343 | 5,6 |
| paired_bootstrap_counts (1000) | 0,190 | 3,1 |
| final_score (build_curves отобранных) | 0,154 | 2,5 |
| sweep_groups / apply_thresholds | 0,034 / 0,034 | 0,6 / 0,6 |

Чтение JSON (`json.loads` стека) в эти цифры не входит и не мерилось отдельно — см. «Идеи».

### 1.2 Внутри build_curves (исходный код, cProfile, `prof/cprofile_build_curves.txt`, всего 7,55 с под профайлером)
| Место | cum, с | % |
|---|---|---|
| официальная `_extract_geometry` (80 315 вызовов) → `shapely.Polygon` по одному | 4,06 | 54 |
|  — из них `Polygon._coords` (2,26 млн точек в Python) | 1,08 | 14 |
|  — `is_valid_scalar` | 0,37 | 5 |
| собственное тело `_collect` (циклы по объектам) | 1,23 | 16 |
| `shapely.transform` + pyproj (3857 → UTM) | 0,73 / 0,54 | 10 / 7 |
| `_geom_max_abs_x` по одному | 0,34 | 4,5 |
| to_wkb + хэш для кэша /dev/shm | 0,15 + 0,07 | 3 |

Дополнительно без профайлера (`tools/time_curves.py`, NOGC=1): исходный build_curves 3,73 с → 3,34 с при
выключенном GC, т.е. **сборщик мусора ≈ 10 % исходного и ≈ 26–30 % уже ускоренного кода** (каждая аллокация
запускает обход огромного графа объектов GeoJSON).

После правок профиль ровный (py-spy, `prof/pyspy_curves_new2.txt`, `tools/spy_top.py`): extract_many 10 %,
to_wkb 8,3 %, is_valid 7,1 %, shapely.transform 5,5 %, intersection (сверка) 4,7 %, batch hash 4,7 %,
pyproj 3,4 % … — горячей точки >10 % больше нет.

### 1.3 Прочее из задания
- `materials/compute_metrics.py` — официальный файл организаторов, менять нельзя; его дорогая часть та же
  `_extract_geometry` (её и заменяет быстрый путь — с проверкой побайтного совпадения).
- post/fusion/candidates/centers/writer — новых замеров **не делал** (это инференс, нужны растры/модели —
  «никаких тяжёлых инференсов»). Есть чужой замер `notes/profile_mix.log` (219,8 с): torch `.cpu()` 103,8 с
  (ожидание GPU, не CPU), probs_to_features 40,2, mask_to_polygons 21,5 (rasterio shapes 10,8), band_stats 11,5,
  object_polys 11,4, split_by_peaks 8,9 с. `notes/etgr_profile_2026-10-22.json` — не профиль CPU;
  `notes/bench_gtx1650.json` — только GPU.

## 2. Оценка vs замер по горячим точкам
Оценки — до работы, по профилю (это **оценки**). Замеры — медианы, файлы в `bench/`.

| Горячая точка | Оценка | Замер | Медианы до → после, с | Источник |
|---|---|---|---|---|
| геометрия из GeoJSON (`_extract_geometry` ×80 315) | ×5–10 | **×4,85** | 1,788 → 0,369 (5 повт.) | bench/extract.json, tools/bench_extract.py |
| build_curves целиком (+GC, +векторный max_abs_x, +потоки PROJ) | ×2–3 | **×2,19** | 3,764 → 1,721 (6 повт.) | bench/curves_{old,new}_*.json, tools/time_curves.py |
| honest_counts, 69 тыс. | ×5 | **×6,0** | 0,444 → 0,074 (5) | bench/stages_x1.json |
| honest_counts, ×20 = 1,39 млн | ×5–10 | **×9,9** | 4,548 → 0,461 (3) | bench/stages_x20.json |
| bootstrap(200), ×1 / ×20 | ×10 | **×12,4 / ×15,2** | 1,964 → 0,159 / 32,90 → 2,17 | stages_x1 / x20 |
| paired_bootstrap(1000), ×1 / ×20 | ×10 | **×16,8 / ×63** | 0,313 → 0,019 / 3,836 → 0,061 | stages_x1 / x20 |
| paired_bootstrap_counts(1000) | ×10 | **×20,8 / ×20,4** | 0,194 → 0,009 / 0,170 → 0,008 | stages_x1 / x20 |
| sweep_groups | ×2 | ×3,2 / ×4,6 | 0,020 → 0,006 / 0,192 → 0,042 | stages_x1 / x20 |
| Весь harness (все шаги scoreboard, без JSON) | ×1,5–2 | **×1,67** | 6,201 → 3,716 (3 серии) | bench/harness_{old,new}_*.json |

«×20» — реальные кривые выборки, размноженные 20 раз (повтор + устойчивая пересортировка, n_gt ×20) —
объём как у полного стека (1 386 720 предсказаний); геометрию на полном стеке не гонял (лимит 2 ГБ ОЗУ).
Итог harness ×1,67 упирается в build_curves: его ускорение ×2,2, остальное ×10+. Внутри harness
build_curves (3,18 с) медленнее отдельного замера (1,72 с): первый вызов, загрузка официального модуля,
запись кэша /dev/shm (to_wkb+хэш) — то же и у исходника. Результаты harness «до» и «после»
(in-sample 0,066208, honest 0,051314, pb, pbc, hits) **совпадают побайтно** (JSON без times).

## 3. Что сделано и где
| Файл (в `bot-8/repo`) | Что |
|---|---|
| `rust/expds_core/src/geo.rs` (новый) | обход GeoJSON-полигонов прямо по PyList/PyFloat → плоские массивы. Быстрый путь — только «простые»: тип ровно `"Polygon"`, кольца — списки/кортежи пар конечных float/int (не bool), ≥4 точек после замыкания (замыкание как в GEOS). Всё прочее — «не взято» |
| `rust/expds_core/src/lib.rs` | `flatten_polygons(list) -> (coords N×2, ring_off, geom_off, taken)` |
| `expds/fastgeom.py` (новый) | `extract_many`: Rust → `shapely.from_ragged_array` → `is_valid` (кусками в потоках); валидные берутся, **всё невалидное, MultiPolygon, Point, 3D, пустое — через официальную `_extract_geometry`**. `max_abs_x` (векторно, та же семантика max() с NaN). `utm_coords_fn` — тот же PROJ кусками в потоках (свой Transformer на поток). Фолбэк: нет модуля/функции, `EXPDS_FASTGEOM=0` или `EXPDS_BACKEND=numpy` → старый код |
| `expds/metric.py` | `_collect` = gc.disable() + `_collect_impl` (быстрый путь), исходник сохранён как `_collect_reference`. Векторные `sweep_groups`, `honest_counts`, `bootstrap`, `paired_bootstrap`, `paired_bootstrap_counts`; исходные — `*_reference`. `EXPDS_FAST_METRIC=0` → исходные. Кэш нормализации классов только для str (у 1/True/1.0 одинаковый хэш) |
| `inference/lib/expds_core.abi3.so` | пересобран (manylinux2014 через `maturin --zig`, максимум GLIBC_2.14 — как у прежнего; target-cpu=x86-64-v3). Старый — `prof/expds_core.abi3.so.orig`. Прежние функции крейта проверены всем набором тестов |
| `tests/test_bot8_fast_metric.py` (новый) | тесты равенства (ниже) |
| `bot-8/tools/*.py`, `bot-8/bench/*`, `bot-8/prof/*` | замеры и профили |

Почему результат тот же: TP/FP/n_gt и веса бутстрепа — целые, суммы/матричные произведения в float64 точны
в любом порядке (< 2^53); затем та же формула F1 (`_f1`) над теми же числами, num/den копятся в порядке
классов; ГСЧ вызывается в том же порядке (`rng.integers` на итерацию + bincount). Точки в PROJ независимы →
кусками тот же результат. Геометрия, которую не удаётся построить побайтно так же, уходит в официальную функцию.

## 4. Тесты равенства и допуски
`cd repo && BOT8_REAL=1 EXPDS_METRIC=$PWD/materials/compute_metrics.py python -m pytest tests/test_bot8_fast_metric.py tests/test_metric_exact.py tests/test_honest.py`
→ **64 passed** (50 моих + 14 существующих; при `EXPDS_METRIC` скипов нет; `prof/equality_tests.txt`). **Допуск везде нулевой**:
- геометрия: `to_wkb` байт в байт + тип; None ↔ None; 37 краевых случаев (незамкнутые кольца, дырки, бабочка,
  3-точечные, вырожденные/нулевой площади, одна точка, int/bool/str/NaN/±1e300, кортежи, np.float64, 3D,
  пустые и битые coordinates, MultiPolygon с равными площадями, Point, LineString, None, {}) + 3000 случайных
  «звёзд» (>100 невалидных → make_valid). Тест нашёл реальную ошибку: NaN в кольце ломал `from_ragged_array`
  (GEOS) — исправлено (не конечные числа → официальный путь);
- `_collect` vs `_collect_reference` (WKB, conf с NaN, регион, группа, порядок) — синтетика 3857 с мусорными
  confidence/классами, пустые коллекции;
- `build_curves` быстрый vs исходный: синтетика, пустые предсказания / пустой эталон / оба пустые,
  реальная выборка 12 участков (41 440 пред. / 534 GT), реальный поток scoreboard на 69 336 / 10 979
  (BOT8_REAL=1): conf/hit побайтно, n_gt, gt_regions с порядком;
- `sweep_groups` (max/plateau), `honest_counts` (drop/keep, k=3/10/по участку, без фолдов, пустые кривые,
  группа только на одном участке, участки без предсказаний, класс без эталона, много равных confidence),
  `bootstrap`, `paired_bootstrap` (n=0/1/50/300, класс только в одной модели, пустые),
  `paired_bootstrap_counts` (regions=None/подмножество/пустое, n=0, разные семейства): `json.dumps` — равенство
  чисел **и порядка ключей**;
- `utm_coords_fn` кусками в 4 потоках vs один вызов (300 тыс. точек) и `par_is_valid` — `tobytes()`/списки равны;
- весь набор тестов репозитория с новым .so: `prof/full_tests_new.txt` — 311 passed, 9 skipped,
  1 failed (`test_coco_roundtrip`: `No module named 'cv2'`, к метрике не относится).

Известная теоретическая разница: исходный `sweep_groups` хранит группы в NumPy U-строках, которые срезают
хвостовые `'\x00'`; быстрый работает со str. Различие возможно только для имён групп с `\x00` в конце.

## 5. Сборка и запуск на Windows (Python 3.12, 12 потоков, AVX2) — исправлено 28.10 (было 20/8)
```bat
:: 1) Rust: https://rustup.rs → rustup-init.exe (toolchain stable-x86_64-pc-windows-msvc; нужен MSVC Build Tools C++)
rustup default stable
:: 2) Python 3.12 venv
py -3.12 -m venv .venv && .venv\Scripts\activate
pip install maturin numpy shapely pyproj pytest orjson
:: 3) сборка (из корня репо)
cd rust\expds_core
set RUSTFLAGS=-C target-cpu=native
::   либо переносимо на любой AVX2-CPU:  set RUSTFLAGS=-C target-feature=+avx2,+fma
maturin build --release -i python -o ..\..\wheels
cd ..\..
pip install --force-reinstall --no-deps wheels\expds_core-0.1.0-cp312-abi3-win_amd64.whl
:: 4) потоки — 12 на всё: крейт читает EXPDS_THREADS, метрика — EXPDS_METRIC_THREADS,
::    numpy/BLAS — OMP_NUM_THREADS, rayon вне крейта — RAYON_NUM_THREADS
set EXPDS_THREADS=12
set EXPDS_METRIC_THREADS=12
set RAYON_NUM_THREADS=12
set OMP_NUM_THREADS=12
:: 5) файловый кэш геометрий метрики (работает и на Windows; по умолчанию %TEMP%\expds_geom_cache)
set EXPDS_GEOM_CACHE=D:\expds_cache\geom
set EXPDS_GEOM_CACHE_MB=4000
:: 6) проверка
python -c "import expds_core as e; print(e.version(), hasattr(e, 'flatten_polygons'))"
::   ожидается: 0.1.0 avx2=true/true threads=12 True
```
- Тесты на Windows: `pytest` подкладывает `inference\lib` первым в sys.path — там лежит Linux `.so`, Windows его
  не подхватит и возьмёт установленное колесо (.pyd). Проверить: `python -c "import expds_core; print(expds_core.__file__)"`.
- Кэш геометрий теперь файловый (`expds/geomcache.py`, см. REPORT3.md), а не `/dev/shm` → работает на Windows.
  На Windows **не проверялся** (нет машины); тесты конкурентности — только Linux.
- Отключить всё: `EXPDS_FAST_METRIC=0` (пороги/бутстрепы), `EXPDS_FASTGEOM=0` (геометрия), `EXPDS_BACKEND=numpy` (без Rust).
- Linux-сборка здесь: `RUSTFLAGS="-C target-cpu=x86-64-v3" maturin build --release --zig --compatibility manylinux2014 -i python`
  → `wheels_mlx2014/expds_core-0.1.0-cp312-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl`
  (version: `0.1.0 avx2=true/true threads=2`). Для площадки по-прежнему `scripts/build_rust.py linux` (docker).

## 6. Что ещё можно (не сделано)
- Чтение стека: orjson/simdjson вместо `json.loads` (вне замеров выше; оценка ×2–3 на разборе — **не мерил**).
- UTM в Rust — быстрее PROJ, но не побайтно → отказался.
- Точки (Point) и MultiPolygon — тоже в Rust (сейчас ~3,5 тыс. объектов идут официальным путём; оценка −0,05 с).
- `from_ragged_array` / `to_wkb` кусками в потоках (GEOS отпускает GIL) — на 12 потоках Windows даст больше.
- Кэш кривых между запусками scoreboard (pickle по хэшу стека) — убирает build_curves при повторах.
- Инференс по `notes/profile_mix.log`: probs_to_features 40 с, rasterio shapes 10,8, band_stats 11,5,
  object_polys 11,4 — кандидаты в Rust (rayon), но нужен честный замер на CPU-растрах; не делал.
