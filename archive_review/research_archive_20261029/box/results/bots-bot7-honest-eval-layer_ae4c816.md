# koz3 test report: `bots/bot7-honest-eval-layer` @ `ae4c816`

- started: 2026-10-29 16:30:17 MSK
- finished: 2026-10-29 16:31:30 MSK
- resolved: `origin/bots/bot7-honest-eval-layer` (`ae4c816cbcb04d1653afe1a52298197539fd97b5`)
- worktree: `/workspace/koz3-wt/bots-bot7-honest-eval-layer`
- venv: `/workspace/koz3-venv` (CPU-only torch)
- host: box «1» (8 CPU, low free RAM)
- command: pytest -q -rfEs -p no:cacheprovider -p koz3_box_env tests centerhead/tests
- test paths: `tests centerhead/tests` (only dirs that exist on this branch)
- wall time: 71.2s
- peak RSS: 2080.6 MiB
- exit: 1

## Counts

| passed | failed | errors | skipped |
|---:|---:|---:|---:|
| 663 | 11 | 0 | 44 |

## pytest summary lines

```
11 failed, 613 passed, 43 skipped, 71 warnings in 65.21s (0:01:05)
50 passed, 1 skipped, 1 warning in 3.98s
```

## Failures / errors (one line each)

```
FAILED tests/test_bot8_speed_cache.py::test_build_curves_equal_reference_synthetic[0-global]
FAILED tests/test_bot8_speed_cache.py::test_build_curves_equal_reference_synthetic[0-site]
FAILED tests/test_bot8_speed_cache.py::test_build_curves_equal_reference_synthetic[1-global]
FAILED tests/test_bot8_speed_cache.py::test_build_curves_equal_reference_synthetic[1-site]
FAILED tests/test_bot8_speed_cache.py::test_build_curves_equal_reference_synthetic[2-global]
FAILED tests/test_bot8_speed_cache.py::test_build_curves_equal_reference_synthetic[2-site]
FAILED tests/test_bot8_speed_cache.py::test_fast_match_off_equal[global] - Na...
FAILED tests/test_bot8_speed_cache.py::test_fast_match_off_equal[site] - Name...
FAILED tests/test_bot8_speed_cache.py::test_mutated_input_is_not_served_from_cache
FAILED tests/test_bot8_speed_cache.py::test_corrupt_cache_file_does_not_change_curves
FAILED tests/test_bot8_speed_cache.py::test_disabled - NameError: name 'per_s...
```

## Skip reasons (grouped)

```
  8  expds/geo_pixgbm.py is held out of main (EhimenNathan-derived lineage review)
  6  expds/geo_pixgbm.py not in main (EN-similarity hold, docs/MERGE_2026-10-29.md)
  5  submission files not kept in git: models/rerank_arch_lgbm.txt, models/pixgbm/all/poly.txt, models/pixgbm/all/finds.txt, expds/geo_pixgbm.py
  5  нет реальной выборки бота 8
  5  нет EXPDS_METRIC
  2  expds_core без flatten_polygons
  2  нет выборки соседних участков bot-8
  2  нет CUDA
  1  rvt_py wheel is not committed to main (archives excluded by the 29.10 merge policy); restore inference/cnx/wheels/ from origin/bots/bot8-cnx to run the [li_cnx] checker tests
  1  models/rerank_arch_lgbm.txt: LightGBM weights are not kept in git (trained on competition data)
  1  нет LAZ Осечки или эталона подготовки bot-9
  1  rvt_py wheel is not committed to main (archives excluded by the 29.10 merge policy); restore it from origin/bots/bot8-cnx before enabling [li_cnx]
  1  реальный прогон: BOT8_REAL=1 и выборка бота 8
  1  нет файла
  1  старый expds_core
  1  нет официального compute_metrics.py
  1  tests/test_bot7_candidate_v2.py: box env: submission files not in git: models/pixgbm/all/poly.txt, models/pixgbm/all/finds.txt, models/rerank_arch_lgbm.txt, expds/geo_pixgbm.py
TOTAL 44
```

## Notes

- GPU/CUDA / missing submission weights / optional deps → skipped via `koz3_box_env` plugin (not counted as regressions).
- Only existing dirs among `tests/`, `centerhead/tests` are passed to pytest (no exit-4 on missing centerhead).
- Official metric (`EXPDS_METRIC`) and real competition samples are not on this box.
- Full log: `/home/box/shared/koz3/box/results/bots-bot7-honest-eval-layer_ae4c816.log`

Re-run: `run_tests.full.sh bots/bot7-honest-eval-layer`
Cleanup worktree: `run_tests.full.sh --cleanup bots/bot7-honest-eval-layer`
