# koz3 test report: `bots/bot4-large-objects` @ `5097c4a`

- started: 2026-10-29 17:49:18 MSK
- finished: 2026-10-29 17:50:40 MSK
- resolved: `origin/bots/bot4-large-objects` (`5097c4afb675f4400b11e874507d6d8008664225`)
- worktree: `/workspace/koz3-wt/bots-bot4-large-objects`
- venv: `/workspace/koz3-venv` (CPU-only torch)
- host: box «1» (8 CPU, low free RAM)
- command: pytest -q -rfEs -p no:cacheprovider -p koz3_box_env tests
- test paths: `tests` (only dirs that exist on this branch)
- wall time: 80.6s
- peak RSS: 1963.1 MiB
- exit: 0

## Counts

| passed | failed | errors | skipped |
|---:|---:|---:|---:|
| 542 | 0 | 0 | 37 |

## pytest summary lines

```
542 passed, 37 skipped, 22 warnings in 79.01s (0:01:19)
```

## Failures / errors

none

## Skip reasons (grouped)

```
  9  нет вендора transformers: /workspace/koz3_shared/bot-7/work6/vendor_test/lib
  6  нет опорного входа смоука ORGATE (вне git, п. 13.11 ТР)
  5  нет EXPDS_METRIC
  3  tests/test_bot7_candidate_v2.py: box env: submission files not in git: models/pixgbm/all/poly.txt, models/pixgbm/all/finds.txt
  2  нет CUDA
  2  tests/test_bot5_gor_dense.py: box env: submission files not in git: models/pixgbm/all/poly.txt, models/pixgbm/all/finds.txt
  2  tests/test_bot6_orgate.py: box env: submission files not in git: models/pixgbm/all/poly.txt, models/pixgbm/all/finds.txt
  2  tests/test_bot7_v3_flags.py: box env: submission files not in git: models/pixgbm/all/poly.txt, models/pixgbm/all/finds.txt
  1  нет артефактов бокса bot-5 (work/gd/root, analysis/gor_dense)
  1  нет трейна 008_Герите на машине
  1  нет опорного входа смоука ORGATE (вне git)
  1  нет распакованного вендора transformers 5.21.0
  1  tests/test_dfine_aeor.py: box env: submission files not in git: models/pixgbm/all/poly.txt, models/pixgbm/all/finds.txt
  1  tests/test_open_hooks.py: box env: submission files not in git: models/pixgbm/all/poly.txt, models/pixgbm/all/finds.txt
TOTAL 37
```

## Notes

- GPU/CUDA / missing submission weights / optional deps → skipped via `koz3_box_env` plugin (not counted as regressions).
- Only existing dirs among `tests/`, `centerhead/tests` are passed to pytest (no exit-4 on missing centerhead).
- Official metric (`EXPDS_METRIC`) and real competition samples are not on this box.
- Full log: `/home/box/shared/koz3/box/results/bots-bot4-large-objects_5097c4a.log`

Re-run: `./run_tests.sh bots/bot4-large-objects`
Cleanup worktree: `./run_tests.sh --cleanup bots/bot4-large-objects`
