# koz3 test report: `bots/bot6-pixgbm-cleanroom` @ `0632c19`

- started: 2026-10-29 16:17:09 MSK
- finished: 2026-10-29 16:18:52 MSK
- resolved: `origin/bots/bot6-pixgbm-cleanroom` (`0632c199aa9488ac57b866e5d0ec7d40c50340ee`)
- worktree: `/workspace/koz3-wt/bots-bot6-pixgbm-cleanroom`
- venv: `/workspace/koz3-venv` (CPU-only torch)
- host: box «1» (8 CPU, low free RAM)
- command: pytest -q -rs -p no:cacheprovider  (+ centerhead/tests separately)
- wall time: 75.6s
- peak RSS: 2149.6 MiB
- exit: 0

## Counts

| passed | failed | errors | skipped |
|---:|---:|---:|---:|
| 1338 | 0 | 0 | 88 |

## pytest summary lines

```
619 passed, 43 skipped, 71 warnings in 69.04s (0:01:09)
619 passed, 43 skipped, 71 warnings in 69.33s (0:01:09)
50 passed, 1 skipped, 1 warning in 4.08s
50 passed, 1 skipped, 1 warning in 4.27s
```

## Failures / errors

none

## Skip reasons (grouped)

```
 16  expds/geo_pixgbm.py is held out of main (EhimenNathan-derived lineage review)
 12  expds/geo_pixgbm.py not in main (EN-similarity hold, docs/MERGE_2026-10-29.md)
 10  submission files not kept in git: models/rerank_arch_lgbm.txt, models/pixgbm/all/poly.txt, models/pixgbm/all/finds.txt, expds/geo_pixgbm.py
 10  нет реальной выборки бота 8
 10  нет EXPDS_METRIC
  4  expds_core без flatten_polygons
  4  нет выборки соседних участков bot-8
  4  нет CUDA
  2  rvt_py wheel is not committed to main (archives excluded by the 29.10 merge policy); restore inference/cnx/wheels/ from origin/bots/bot8-cnx to run the [li_cnx] checker tests
  2  expds/geo_pixgbm.py отсутствует на этой ветке (gitignore main)
  2  models/rerank_arch_lgbm.txt: LightGBM weights are not kept in git (trained on competition data)
  2  нет LAZ Осечки или эталона подготовки bot-9
  2  rvt_py wheel is not committed to main (archives excluded by the 29.10 merge policy); restore it from origin/bots/bot8-cnx before enabling [li_cnx]
  2  реальный прогон: BOT8_REAL=1 и выборка бота 8
  2  нет файла
  2  старый expds_core
  2  нет официального compute_metrics.py
TOTAL 88
```

## Notes

- GPU/CUDA tests are skipped on this runner (CPU-only).
- Missing submission files (pixgbm / rerank weights / geo_pixgbm.py) cause intentional skips.
- Official metric (`EXPDS_METRIC`) and real competition samples are not on this box.
- Full log: `/home/box/shared/koz3/box/results/bots-bot6-pixgbm-cleanroom_0632c19.log`

Re-run: `/home/box/shared/koz3/box/run_tests.sh bots/bot6-pixgbm-cleanroom`
Cleanup worktree: `/home/box/shared/koz3/box/run_tests.sh --cleanup bots/bot6-pixgbm-cleanroom`
