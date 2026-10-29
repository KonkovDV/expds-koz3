# koz3 test report: `main` @ `70dedf4`

- started: 2026-10-29 16:10:30 MSK (approx)
- finished: 2026-10-29 16:11:43 MSK
- resolved: `origin/main` (`70dedf49b0fa7013718885d466fb9fcaab5f3c00`)
- worktree: `/workspace/koz3-wt/main` (cleaned after run)
- venv: `/workspace/koz3-venv` (CPU-only torch 2.11.0+cpu)
- host: box «1» (8 CPU, ~2 GB free RAM shared)
- command: pytest -q -rs -p no:cacheprovider  (+ centerhead/tests separately)
- wall time: 72.3s
- peak RSS: 2179.9 MiB
- exit: 0

## Counts

| passed | failed | errors | skipped |
|---:|---:|---:|---:|
| 666 | 0 | 0 | 43 |

(tests/: 616 passed, 42 skipped; centerhead/tests: 50 passed, 1 skipped)

## pytest summary lines

```
616 passed, 42 skipped, 71 warnings in 65.92s (0:01:05)
50 passed, 1 skipped, 1 warning in 4.17s
```

## Failures / errors

none

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
TOTAL 43
```

## Notes

- GPU/CUDA tests are skipped on this runner (CPU-only): 2 skips «нет CUDA».
- Missing submission files (pixgbm / rerank weights / geo_pixgbm.py) cause intentional skips.
- Official metric (`EXPDS_METRIC`) and real competition samples are not on this box.
- `expds_core` built with rustup rustc 1.99.1 (system 1.85.1 too old for delaunator/uuid); CI itself does not build Rust.
- Peak RSS ~2.2 GiB — запускать **по одной** задаче.
- Full log: `/home/box/shared/koz3/box/results/main_70dedf4.log`

Re-run: `/home/box/shared/koz3/box/run_tests.sh main`
Cleanup worktree: `/home/box/shared/koz3/box/run_tests.sh --cleanup main`
