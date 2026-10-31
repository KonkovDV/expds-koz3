# REPORT3 — bot-8: ускорение build_curves, файловый кэш геометрий, слияние веток, архив, Windows 12 потоков
28.10.2026, 22:15 МСК. Без LB, без push. Сопоставление по умолчанию — **global** (как официальная метрика); site — только диагностика.

## 1. Скорость: весь поток scoreboard --honest (цель ≥ ×5 — НЕ достигнута)
Поток `tools/scoreboard_flow.py` (data/stack_every20 = каждое 20-е предсказание стека, 69 336 шт. + gt_train_v2;
evaluate база/опыт, honest 10 ×2, paired_bootstrap 1000, paired_bootstrap_counts 1000), 2 потока, nice 10.
По 3 запуска, total_flow, с (файлы `bench/r3/*.json|log`); выходные JSON (без времён) равны во всех 18 запусках.

| версия | холодный кэш (3 запуска) | медиана | тёплый кэш (3 запуска) | медиана |
|---|---|---|---|---|
| main 24bd916 | 8,006 / 8,596 / 9,126 | 8,596 | 4,774 / 5,334 / 5,093 | 5,093 |
| прежний bot-8 bd39adc | 5,887 / 5,221 / 4,613 | 5,221 | 3,834 / 3,578 / 3,538 | 3,578 |
| новый (3745997 → bot8-merged) | 3,770 / 3,347 / 3,371 | 3,371 | 2,497 / 2,103 / 2,144 | 2,144 |

- Новый к main: **×2,55** холодный, **×2,38** тёплый (main холодный → новый тёплый ×4,0). Цель ×5 не взята.
- Шум: разброс внутри версии ≈ ±10 % (main холодный 8,0–9,1 с) — различия версий много больше шума.
- Что держит: чтение JSON 0,65–0,75 с даже с orjson; Python-цикл `_collect` (20 вызовов на 10 build_curves,
  `prof/cp_warm3.prof`); холодная запись кэша ~0,3 с/файл (проверка обратимости). Стадии тёплого build_curves на
  every20: flatten 0,12, ключ 0,05, чтение кэша 0,127, collect 0,468, всего 0,734 с.
- Что сделано: векторное качество пар (охрана sha256 исходников официальных функций), память пар, STRtree эталона,
  orjson с защитой больших целых (Rust `has_digit_run` ≥19 цифр → json), файловый кэш геометрий.

## 2. Равенство (байт в байт), обе области
- `bench/eq3_{sub,every20,cross}.json|log`: кривые быстрые == эталонные (прежний код) в **global и site**, холодно,
  с файловым кэшем и с памятью пар — equal=True везде.
- `tests/test_bot8_speed_cache.py` (38 тестов): равенство с эталоном в обеих областях (синтетика, реальные sub/cross),
  путь EXPDS_FAST_MATCH=0, мутация входа, смешанные типы геометрий, пустые партии, битые/обрезанные файлы кэша
  (кривые не меняются), ключ зависит от содержимого и параметров, выключенный кэш, чистка по размеру,
  конкурентность потоками и процессами (spawn), orjson == json на краевых случаях.
- `EXPDS_FAST_METRIC=0` — весь прежний путь (build_curves → build_curves_reference, без orjson/кэша/памяти пар).

## 3. Кэш геометрий (`expds/geomcache.py`, формат v2)
- Ключ: sha256(отпечаток окружения: версия формата, sha256 файлов metric.py/fastgeom.py/geomcache.py,
  sha256 официального скрипта, версии shapely+GEOS и pyproj+PROJ, порог UTM; + параметры партии: режим, is_gt, r_tol, c_pts) + **полное содержимое** партии (выход Rust flatten + repr прочего). Любое изменение → промах.
- Хранение: полигоны/точки — рваные массивы координат, прочее — WKB; если хоть одна геометрия не проходит обратную
  проверку WKB — вся партия в WKB. Целостность blake2b-16; запись атомарная (tmp + os.replace) — безопасно при
  параллельных процессах; повреждённый файл = промах. Чистка до `EXPDS_GEOM_CACHE_MB`.
- Каталог: `EXPDS_GEOM_CACHE=<dir>` | `off`; по умолчанию Linux `/dev/shm/expds_utm_cache`, **Windows
  `%TEMP%\expds_geom_cache`** (файловый, работает без /dev/shm). На этой машине /dev/shm не используем — задавать каталог.
- Минимальная партия `EXPDS_GEOM_CACHE_MIN` (200).

## 4. Ветки и коммиты (локально, без push; репо `/workspace/koz3_shared/bot-8/repo`)
**bot8-merged** = 2f530a8:
- 9316b5f, 8343483 — устойчивость (bot8-reliability);
- a49ad85 — Rust flatten_polygons + numpy (быстрый сбор геометрий);
- bd39adc — scope global|site, score_by_site;
- 3745997 — ускорение build_curves, файловый кэш, orjson, тесты;
- 50228d9 — site только явно + предупреждение в stderr, тест «по умолчанию global»; scripts/stack_scoreboard.py,
  scripts/stack_official_kept.py (REPORT4);
- 2f530a8 — merge свежей bot8-reliability (2901cb0, дедлайн в цикле окон снимков — коммит другого агента).
Незакоммиченного нет (`git status` чист). Диф: `patch/bot8-merged.diff` (24bd916..bot8-merged, 19 файлов, +3231/−31, с .so).

**bot8-rust-metric** = d073b45 (для bot-10; от main 24bd916, без устойчивости): 6911024 (= a49ad85), ba2e9a4 (= bd39adc),
d073b45 (site только явно + тест). Диф: `patch/bot8-rust-metric.diff`. Ускорение 3745997 туда не входит.

Тесты:
- bot8-merged 2f530a8: **414 passed, 10 skipped, 1 failed** (`bench/pytest_full5.txt`); падает
  test_coco_roundtrip — нет модуля cv2 в venv (окружение, не код).
- bot8-rust-metric (2 коммита): 364 passed, 10 skipped, 1 failed тот же cv2 (`bench/pytest_rust_metric_branch.txt`);
  после d073b45 — подмножество метрики 60 passed, 6 skipped (`bench/pytest_rust_metric_branch_subset.txt`).

## 5. Архив submission_crop (воспроизводимый, из git-коммита)
`./build_submission.sh submission_crop --ref bot8-merged` (сборка + check_submission.py), второй раз —
`build_submission.py ... --out-dir /tmp/bot8_dist2` (удалён после сравнения):
- `dist/koz3_submission_crop_2f530a8_1e8f8e68.zip`, 538,6 МиБ, 46 файлов;
- sha256 обеих сборок: **dcad2ddd76492a797b2258af4caa49dd4a3a92b868fe3e90721f4358c294da54** (одинаковы, `bench/submission_crop_sha256.txt`);
- check_submission.py: **PASS — FAIL 0, WARN 0, 26 проверок**; smoke без сети 16 с, пик RSS 972 МиБ (`bench/build_submission_crop_1.log`).
  Манифесты двух сборок отличаются (путь вывода/время), zip — побайтно одинаковы.
- Это архив кода bot8-merged; он не тот же, что dist/submission_crop.zip у bot-3 (bd52ad…b774).

## 6. Windows, 12 потоков (исправлен и RUST_REPORT.md §5: было 20/8)
```bat
set EXPDS_THREADS=12
set EXPDS_METRIC_THREADS=12
set RAYON_NUM_THREADS=12
set OMP_NUM_THREADS=12
set EXPDS_GEOM_CACHE=D:\expds_cache\geom
set EXPDS_GEOM_CACHE_MB=4000
:: EXPDS_MATCH_SCOPE не задавать (= global)
python -c "import expds_core as e; print(e.version())"   :: ожидается ... threads=12
python scripts\scoreboard.py --work ... --train ... --honest 10 --bootstrap 1000 --out res.json
```
Сборка колеса — RUST_REPORT.md §5 (maturin, target-cpu=native, `pip install orjson`).

## 7. Что не проверено
- Windows вообще (нет машины): сборка колеса, файловый кэш на NTFS, конкурентность процессов на Windows, 12 потоков.
- Скорость на 12 потоках — не мерил (здесь ≤2).
- Полный scoreboard.py с `--work/--train` (нет LAS трейна на этой машине) — мерил эквивалентный поток scoreboard_flow.
- Цель ×5 не достигнута (см. §1).
- old_wt (бенч-дерево bd39adc) удалён `git worktree remove`; wt-rel* — деревья другого агента, не трогал.
