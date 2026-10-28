# bot-8 → bot-7: исправления надёжности поверх bot7-bundle

Обновлено: 2026-10-28 ~22:10 MSK. Ничего не пушилось и не отправлялось на LB.

## Где нашёл bot7-bundle
`/workspace/koz3_shared/bot-7/repo`, ветка `bot7-bundle` = `42ada8d` (на main `24bd916`).
Там же `bot7-candidate` = `fe0e570` (уже содержит наш 8343483 как `07bed29` + micro_circles + rerank_arch),
поэтому сделал две ветки: под bundle и под candidate. Ветки bot-7 подтянуты в `bot-8/repo` только как
`refs/remotes/bot7/*` (`git fetch`, рабочее дерево не трогалось).

## Ветки (в `/workspace/koz3_shared/bot-8/repo`, рабочие деревья — рядом)
| ветка | база | коммиты bot-8 | дерево |
|---|---|---|---|
| `bot8-reliability-on-bot7` | `42ada8d` bot7-bundle (← 24bd916) | f06776f, ca38bee, 9ae4c33, 68bd7a7, **5135517** | `bot-8/wt-rel7` |
| `bot8-reliability-on-bot7-candidate` | `fe0e570` bot7-candidate | 2dd45fa, 2621e33, 7a9f1d1, **fd7dd50** | `bot-8/wt-rel7c` |
| `bot8-reliability` (исходная линия) | `24bd916` main | 9316b5f, 8343483, **2901cb0** | `bot-8/wt-rel` |

Полные хэши (bundle): f06776fb938a2effa73067720dc44c769383a100, ca38bee74554ff73444aa40e567cdb2562d80b59,
9ae4c3346d0df94a72801583a2d5a36392b966a9, 68bd7a75bf0d31c2969e53bbf5a8e7a09c40417a, 5135517ad6f91ddce80dc4e0fc4f36eb1018f984.
Жирным — новый отдельный коммит «дедлайн внутри цикла окон» (задача от бота 1).

## Как применить
Серии патчей (`git format-patch`), проверено `git am -3` на чистых worktree: дерево после am совпадает с веткой
(tree d3cd79a… / dc59c88… / 163a587…):
```bash
# на bot7-bundle
git checkout bot7-bundle && git am -3 /workspace/koz3_shared/bot-8/patch/for-bot7/0*.patch
# на bot7-candidate (без 0002 — он уже есть как 07bed29)
git checkout bot7-candidate && git am -3 /workspace/koz3_shared/bot-8/patch/for-bot7-candidate/0*.patch
# или cherry-pick из общего объекта:
git fetch /workspace/koz3_shared/bot-8/repo bot8-reliability-on-bot7 && git cherry-pick 42ada8d..FETCH_HEAD
git fetch /workspace/koz3_shared/bot-8/repo bot8-reliability-on-bot7-candidate && git cherry-pick fe0e570..FETCH_HEAD
```
Только дедлайн: `patch/for-bot7/0005-bot-8.patch` (= `for-bot7-candidate/0004-bot-8.patch`), применяется отдельно.
Сводные diff: `patch/for-bot7/bot8-reliability-on-bot7.diff`, `patch/for-bot7-candidate/bot8-reliability-on-bot7-candidate.diff`,
`patch/bot8-reliability.diff` (перегенерирован: main..2901cb0, теперь с дедлайном), серия `patch/for-bot8-reliability/`.

## Конфликты и как решены
1. bot7-bundle + 9316b5f/8343483: текстовых конфликтов нет. Смысловой: запасная цепочка `safe_finish`
   (срабатывает, если `finish()` упал) не знала шагов bot-7 → ответ в аварийном пути шёл бы без микрокругов
   (и без rerank_arch на candidate), т.е. с другой геометрией/порогами, чем обычный путь. Решение — коммит 9ae4c33/2621e33:
   в запасной цепочке тот же порядок, что в `finish` bot-7: only_known_classes → rerank_arch.apply (если enabled) →
   apply_thresholds → `out + reranked` → post.micro_circles (если задан) → finalize по объекту → cap → only_classes.
   Каждый шаг bot-7 в своём try: его сбой не обнуляет ответ. Поведение bot-7 в обычном пути не меняется.
2. bot7-candidate: 8343483 пропущен (это уже 07bed29, + tests/test_bot8_crop_root.py). Конфликт в
   `tests/test_bot8_faults.py` при переносе 9ae4c33: убрал дубль теста verify (покрыт test_bot8_crop_root.py), добавил
   `test_safe_finish_fallback_keeps_bot7_rerank_arch` (хелперы из test_bot7_rerank_arch).
3. Инструменты сдачи (68bd7a7/7a9f1d1): `scripts/check_submission.py` и `build_submission_git.py` учитывают
   `[rerank_arch] model/meta` (без этих файлов rerank молча возвращает базу) — проверка путей и загрузка бустера.
4. Дедлайн (5135517/fd7dd50/2901cb0): cherry-pick без конфликтов во все три линии. Подписи вызывающих
   (`optical_feats`, `geo_model_feats`, `_wide_without_lidar`) не менял — первая версия с аргументом `guard=` ломала
   4 существующих теста с подделками `_raster_with_model` (test_pipeline ×2, test_geo_nn) — переделал на `_WINDOW_GUARD`.

## От чего защищает каждое исправление (строки — ветка bot8-reliability-on-bot7, 5135517)
- `expds/pipeline.py:928-933` — try вокруг `geophys_feats` в цикле регионов: сбой геофизики одного региона больше
  не обрывает весь прогон (раньше исключение выходило из try run() → лидар/оптика всех регионов пропускались).
  Тест: `tests/test_bot8_faults.py:17`.
- `expds/watchdog.py:104-140` `safe_finish`, используется в `checkpoint` (`watchdog.py:188`): исключение в
  постобработке (GEOS в склейке и т.п.) не оставляет пустой/недописанный ответ; сохраняет шаги bot-7.
  Тесты: `test_bot8_faults.py:50`, `:99` (micro_circles), на candidate ещё rerank_arch.
- `expds/writer.py:29-30, 63` — объекты с NaN/inf координатами отбрасываются, `fid` подряд, `allow_nan=False`:
  без этого в файл попадает литерал `NaN` → невалидный JSON → ноль на LB. Тест: `test_bot8_faults.py:70`.
- `expds/pipeline.py:124` `rooted_verify` (вызов `:1009`): `[verify] weights` от корня архива, а не от cwd
  (на площадке cwd ≠ корень → проверка кропа тихо выключалась). Тест: `test_bot8_faults.py:86`.
- **Дедлайн в цикле окон** — `expds/pipeline.py:31-82` (TimeGuard.window_left/window_ok, `_WINDOW_GUARD`,
  `window_reserve`), `:760-761`, `:796-803` (проверка перед каждым окном в `_raster_with_model`), `:855-861`
  (guard run() с `hard_left = watchdog.deadline_left(guard_margin_s)`), `:973-976` (`state.note["partial_files"]`),
  `:999` (сброс), `:1018` (`summary["partial_files"]`); `expds/watchdog.py:246` `deadline_left` (тот же расчёт, что
  у `arm_platform`: 1800 − 180 = 1620 с от `PROCESS_START`, новых часов нет), `:171-173` строка в протоколе;
  `configs/default.toml:10-14`: `window_deadline = true`, `window_reserve_s = 30`.
  Раньше бюджет проверялся только между файлами/регионами (`guard.ok(60)`): один большой растр мог съесть остаток
  до сторожа. Теперь окно не начинается, если до min(конец budget_s, сторож 1620 с) ≤ window_reserve_s; файл отдаёт
  объекты пройденных окон, дальше обычный конец: канал σ/verify сами пропускаются по `guard.ok`, `checkpoint` →
  `safe_finish` → `writer.write_geojson`. Рецепт: competitors/NOTES_KateBuikina-expds-solution.md §5
  (`segmenter._deadline_hit` в начале цикла окон). Вне `run()` (scripts/geo_predict, nn_oof) проверки нет.
  Ограничение: окно 4096 px — на маленьких растрах это 1 окно, тогда частичность = «следующий файл не начат».
  Тесты `tests/test_bot8_window_deadline.py` (поддельные часы, каждое окно «длится» 100 с):
  `:43` частичный результат (3 окна из N, подмножество полного), `:60` выключено/далеко — полный результат,
  `:71` конфиг, `:77` сторож ближе бюджета решает он, `:88` весь `run()`: файл валиден (`writer.validate`),
  объектов 0 < n < 7, `summary["partial_files"]` = 3 окна, `os._exit` не вызывался, `_WINDOW_GUARD` сброшен.

## Тесты (OMP 2 потока, nice 10; логи `bot-8/work/rel7/`)
| дерево | результат | лог |
|---|---|---|
| main 24bd916 | 1 failed, 307 passed, 9 skipped | tests_wt-base-main.log |
| bot7-bundle 42ada8d | 1 failed, 307 passed, 9 skipped | tests_wt-base-b7.log |
| bot8-reliability 2901cb0 | 1 failed, 316 passed, 9 skipped | tests_wt-rel.log |
| **bot8-reliability-on-bot7 5135517** | 1 failed, 317 passed, 9 skipped | tests_wt-rel7.log |
| bot8-reliability-on-bot7-candidate fd7dd50 | 1 failed, 322 passed, 9 skipped | tests_wt-rel7c.log |

Единственное падение везде одно и то же и есть уже на main: `tests/test_coco_roundtrip.py::test_grid_cache_roundtrip_gives_one_circle_inside_each_object`
(`ModuleNotFoundError: cv2` в venv) — не наше.
Регрессионные тесты bot-8 на чистом bot7-bundle (без исправлений): **10 failed** из 10 (`regr_on_bot7_bundle.log`) —
то есть тесты ловят именно эти дефекты.

## Архив и check_submission
`scripts/build_submission_git.py --repo <worktree> --ref HEAD --label … --check` (файлы из коммита, не из рабочей папки):
- bundle: `bot-8/dist/koz3_rel7_5135517_f854a54c.zip`, 538.8 МиБ, 44 файла, sha256 c91ed132…aa5a —
  **PASS (FAIL 0, WARN 0, 26 проверок)**, смоук на CPU: 80 объектов, 69.4 с (`work/rel7/build_rel7.log`).
- candidate: `bot-8/dist/koz3_rel7c_fd7dd50_2ed3a301.zip`, 539.0 МиБ, 47 файлов, sha256 70e6ac25…6db4 —
  **PASS (FAIL 0, WARN 0, 26 проверок)**, 14 путей к весам в т.ч. rerank_arch, смоук 80 объектов, 78.7 с (`build_rel7c.log`).
Ничего не отправлялось на LB.
