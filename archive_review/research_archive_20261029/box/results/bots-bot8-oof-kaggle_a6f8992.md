# koz3 test report: `bots/bot8-oof-kaggle` @ `a6f8992`

- started: 2026-10-29 16:21:03 MSK
- finished: 2026-10-29 16:22:47 MSK
- resolved: `origin/bots/bot8-oof-kaggle` (`a6f8992abfdbf300a40f624509ab9f2cd630d394`)
- worktree: `/workspace/koz3-wt/bots-bot8-oof-kaggle`
- venv: `/workspace/koz3-venv` (CPU-only torch)
- host: box «1» (8 CPU, low free RAM)
- command: pytest -q -rs -p no:cacheprovider  (+ centerhead/tests separately)
- wall time: 77.4s
- peak RSS: 2284.5 MiB
- exit: 0

## Counts

| passed | failed | errors | skipped |
|---:|---:|---:|---:|
| 671 | 0 | 0 | 43 |

## pytest summary lines

```
621 passed, 42 skipped, 71 warnings in 71.03s (0:01:11)
50 passed, 1 skipped, 1 warning in 4.13s
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

- GPU/CUDA tests are skipped on this runner (CPU-only).
- Missing submission files (pixgbm / rerank weights / geo_pixgbm.py) cause intentional skips.
- Official metric (`EXPDS_METRIC`) and real competition samples are not on this box.
- Full log: `/home/box/shared/koz3/box/results/bots-bot8-oof-kaggle_a6f8992.log`

Re-run: `/home/box/shared/koz3/box/run_tests.sh bots/bot8-oof-kaggle`
Cleanup worktree: `/home/box/shared/koz3/box/run_tests.sh --cleanup bots/bot8-oof-kaggle`
