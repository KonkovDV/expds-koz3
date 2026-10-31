# koz3 test report: `origin/bots/bot6-pixgbm-cleanroom` @ `b0add08`

- started: 2026-10-29 18:18:28 MSK
- finished: 2026-10-29 18:20:14 MSK
- resolved: `origin/bots/bot6-pixgbm-cleanroom` (`b0add08e194853a351d520df19cfb9810c4db983`)
- worktree: `/workspace/koz3-wt/origin-bots-bot6-pixgbm-cleanroom`
- venv: `/workspace/koz3-venv` (CPU-only torch)
- host: box «1» (8 CPU, low free RAM)
- command: pytest -q -rfEs -p no:cacheprovider -p koz3_box_env tests centerhead/tests
- test paths: `tests centerhead/tests` (only dirs that exist on this branch)
- wall time: 104.9s
- peak RSS: 2125.7 MiB
- exit: 0

## Counts

| passed | failed | errors | skipped |
|---:|---:|---:|---:|
| 674 | 0 | 0 | 47 |

## pytest summary lines

```
624 passed, 46 skipped, 71 warnings in 95.68s (0:01:35)
50 passed, 1 skipped, 1 warning in 5.99s
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
  1  expds/geo_pixgbm.py not in git (EN-similarity hold)
  1  expds/geo_pixgbm.py отсутствует на этой ветке (gitignore main)
  1  models/rerank_arch_lgbm.txt: LightGBM weights are not kept in git (trained on competition data)
  1  нет LAZ Осечки или эталона подготовки bot-9
  1  rvt_py wheel is not committed to main (archives excluded by the 29.10 merge policy); restore it from origin/bots/bot8-cnx before enabling [li_cnx]
  1  реальный прогон: BOT8_REAL=1 и выборка бота 8
  1  нет файла
  1  старый expds_core
  1  geo_pixgbm.py absent
  1  нет официального compute_metrics.py
  1  tests/test_bot7_candidate_v2.py: box env: submission files not in git: models/pixgbm/all/poly.txt, models/pixgbm/all/finds.txt, models/rerank_arch_lgbm.txt, expds/geo_pixgbm.py
TOTAL 47
```

## Notes

- GPU/CUDA / missing submission weights / optional deps → skipped via `koz3_box_env` plugin (not counted as regressions).
- Only existing dirs among `tests/`, `centerhead/tests` are passed to pytest (no exit-4 on missing centerhead).
- Official metric (`EXPDS_METRIC`) and real competition samples are not on this box.
- Full log: `/home/box/shared/koz3/box/results/origin-bots-bot6-pixgbm-cleanroom_b0add08.log`

Re-run: `./run_tests.sh origin/bots/bot6-pixgbm-cleanroom`
Cleanup worktree: `./run_tests.sh --cleanup origin/bots/bot6-pixgbm-cleanroom`
