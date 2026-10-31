# CRS_AUDIT.md — поиск бага бейзлайна `crs = region_crs or file_crs`
bot-8, 28.10.2026 (МСК), только чтение. Ветки не менял.

**Баг бейзлайна** (цитата в прошлых NOTES): `prediction.py:13-31` — СК региона (UTM.json) ставится выше СК файла. Пример: в регионе 161 растр в EPSG:3857, а в UTM.json 32637 → объекты уезжают.

## Охват
- `/workspace/expds-koz3`: все локальные и remote-ветки указывают на один коммит **main 24bd916**.
- Копии ботов: bot-3/repo, bot-4/repo, bot-5/repo, bot-8/repo, bot-8/old_wt, bot-10/repo, bot-10/work4/repo_bot8. Ветки: bot3-geophys, bot4-gorfort, bot5-arch-microcircle, bot10-bot8site-check, bot10-honest-bootstrap, bot10-site-scope-provisional, bot10-unseen-keep, bot8-reliability-ro, bot8-merged, bot8-reliability, bot8-rust-metric.
- Сравнивал блобы crs/lidar/raster/tiles/geophys/fusion/register/catalog/hillfort с main. Отличаются только три ветки:
  - bot3-geophys → `geophys.py`, только поля original_crs;
  - bot4-gorfort → `hillfort.py`, только threshold_group;
  - bot5-arch-microcircle → `tiles.py`, в diff нет строк про crs.
  Логика CRS во всех ветках та же, что в main.
- Патчи: bot-2/work/bot-2-kurgan-morph.diff, bot-6/bot6-optics.diff, diff-файлы bot-7/bot-8/bot-10, `*/repo/patches/0001-li-ae-groups-replace-mode.patch`.

## Результат по местам (main 24bd916)
| Место | Логика | Вердикт |
|---|---|---|
| `expds/crs.py:4` (docstring) | порядок «UTM.json → WKT LAS → CRS растра» | описание; на деле для растров действует файл-первый порядок (ниже) |
| `crs.py:157-182 resolve_region_crs` | СК региона. policy `vote`: plausible() → `_overlap_votes` по растрам региона; LAS побеждает только при **строго большем** числе голосов, при ничьей — UTM.json (181-182). Проверяется только `las_files[0]` (164-165) | ⚠️ предпочтение UTM.json над LAS — **намеренное**, смягчено голосованием. Остаточный риск ниже |
| `pipeline.py:123-132 region_crs` | по умолчанию `utm_json`, конфиг ставит `vote` | ок |
| **`lidar.py:161-164` (build_dem), `lidar.py:321-323` (build_dem_mosaic)** | `if crs is None: crs = las_crs(las_path)` — то есть «регион, иначе файл». Вызовы передают crs_region: `pipeline.py:282, 295, 317, 606`, prefetch `:859`. Дальше d.crs идёт в `pipeline.py:201, 216, 240, 298, 374, 672-701`, `hillfort.py:146-157`, `candidates.py:257` | ⚠️ **это и есть шаблон `region or file`, но только для LAS**. Риск закрыт голосованием в resolve_region_crs |
| `pipeline.py:739` (снимки) | CRS файла, регион — запасной | ✅ file-first |
| `fusion.py:70` | file-first | ✅ |
| `geophys.py:124, 541, 595` (запас из `pipeline.py:172/174`) | file-first | ✅ |
| `register.py:227` | file-first | ✅ |
| `tiles.py:208-209` | `usable_raster_crs(declared, ds.bounds, utm) or utm`; `or utm` — только когда СК файла нет или не распознана | ✅ это фолбэк, не баг |
| `crs.py:223-241 usable_raster_crs` | объявленная СК, если она переводится в 3857; иначе угадывание по границам (utm → UTM.json, webmercator, wgs84) | ✅ лечит именно случай «региона 161» |
| `raster.py:50, 173` (ds.crs); `raster.py:185-205 open_metric` (warp от src CRS) | file-first | ✅ |
| `blobs.py:75`; `analysis/probe_peak_circle.py:58` | file-first | ✅ |
| `health.py:106` (declared), `:114` (считает region_crs) | диагностика | ✅ |
| bot-2-kurgan-morph.diff, `scripts/kurgan_morph.py` utm_transformers 76-85 | UTM.json для 3857→UTM площади GT; растры не читает | ✅ только анализ, низкий риск |
| bot6-optics.diff:50-53 | собственный crs GeoJSON | ✅ |
| diff-файлы bot-7/bot-8/bot-10 | UTM только на стороне метрики | ✅ |
| 0001-li-ae-groups-replace-mode.patch | совпадений по crs нет | ✅ |

## Данные (`notes/crs_audit_2026-10-24.json`)
- 54 региона с LAS: у 44 в LAS нет CRS, у 10 есть.
- Конфликтов 8, все одного типа: ETRS89 EPSG:258xx в LAS против WGS84 EPSG:326xx в UTM.json. Регионы 009, 010, 011, 013, 016, 018, 030, 061.
- Голоса во всех конфликтах 0/0, gt_inside одинаковый; `conflicts_where_las_is_right: 0`.
- ETRS89 ≈ WGS84 с точностью < 1 м (оценка) → оба выбора на деле равноценны.

## Вердикт
1. Для растров (снимки, геофизика, тайлы, fusion, register) **бага бейзлайна нет**: порядок file-first, СК региона — только фолбэк.
2. Для LAS СК региона намеренно перекрывает заголовок (`lidar.py:161-164, 321-323`), но выбор идёт через голосование. 8 известных конфликтов безвредны (ETRS89/WGS84).
3. Остаточные риски для частного набора:
   - (a) проверяется только первый LAS региона;
   - (b) при ничьей голосов побеждает UTM.json;
   - (c) регион с несколькими LAS в разных СК;
   - (d) неверный UTM.json при «правдоподобном» заголовке LAS и 0 голосов (нет растров для голосования).
   Дешёвая страховка, без правки кода сейчас: для LAS при 0/0 голосов и правдоподобной СК в заголовке, которая отличается **зоной/датумом сильнее 1 м**, предпочитать заголовок; и проверять все LAS региона, а не только `[0]`.
