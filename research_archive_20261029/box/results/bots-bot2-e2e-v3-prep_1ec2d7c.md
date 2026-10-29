# koz3 test report: `bots/bot2-e2e-v3-prep` @ `1ec2d7c`

- started: 2026-10-29 16:31:31 MSK
- finished: 2026-10-29 16:32:19 MSK
- resolved: `origin/bots/bot2-e2e-v3-prep` (`1ec2d7cec6c66066a2ef88cae9080cbdf7e163df`)
- worktree: `/workspace/koz3-wt/bots-bot2-e2e-v3-prep`
- venv: `/workspace/koz3-venv` (CPU-only torch)
- host: box «1» (8 CPU, low free RAM)
- command: pytest -q -rfEs -p no:cacheprovider -p koz3_box_env tests
- test paths: `tests` (only dirs that exist on this branch)
- wall time: 46.0s
- peak RSS: 2008.0 MiB
- exit: 0

## Counts

| passed | failed | errors | skipped |
|---:|---:|---:|---:|
| 455 | 0 | 0 | 23 |

## pytest summary lines

```
455 passed, 23 skipped, 21 warnings in 44.89s
```

## Failures / errors

none

## Skip reasons (grouped)

```
  9  нет вендора transformers: /workspace/koz3_shared/bot-7/work6/vendor_test/lib
  5  нет EXPDS_METRIC
  3  tests/test_bot7_candidate_v2.py: box env: submission files not in git: models/pixgbm/all/poly.txt, models/pixgbm/all/finds.txt
  2  нет CUDA
  2  tests/test_bot7_v3_flags.py: box env: submission files not in git: models/pixgbm/all/poly.txt, models/pixgbm/all/finds.txt
  1  нет распакованного вендора transformers 5.21.0
  1  tests/test_open_hooks.py: box env: submission files not in git: models/pixgbm/all/poly.txt, models/pixgbm/all/finds.txt
TOTAL 23
```

## Notes

- GPU/CUDA / missing submission weights / optional deps → skipped via `koz3_box_env` plugin (not counted as regressions).
- Only existing dirs among `tests/`, `centerhead/tests` are passed to pytest (no exit-4 on missing centerhead).
- Official metric (`EXPDS_METRIC`) and real competition samples are not on this box.
- Full log: `/home/box/shared/koz3/box/results/bots-bot2-e2e-v3-prep_1ec2d7c.log`

Re-run: `run_tests.full.sh bots/bot2-e2e-v3-prep`
Cleanup worktree: `run_tests.full.sh --cleanup bots/bot2-e2e-v3-prep`
