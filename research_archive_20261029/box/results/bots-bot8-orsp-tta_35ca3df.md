# koz3 test report: `bots/bot8-orsp-tta` @ `35ca3df`

- started: 2026-10-29 16:27:39 MSK
- finished: 2026-10-29 16:28:30 MSK
- resolved: `origin/bots/bot8-orsp-tta` (`35ca3df6c851d6a02d5a1a1eb872f54fe2c43a83`)
- worktree: `/workspace/koz3-wt/bots-bot8-orsp-tta`
- venv: `/workspace/koz3-venv` (CPU-only torch)
- host: box «1» (8 CPU, low free RAM)
- command: pytest -q -rfEs -p no:cacheprovider -p koz3_box_env tests
- test paths: `tests` (only dirs that exist on this branch)
- wall time: 49.3s
- peak RSS: 1880.7 MiB
- exit: 0

## Counts

| passed | failed | errors | skipped |
|---:|---:|---:|---:|
| 424 | 0 | 0 | 7 |

## pytest summary lines

```
424 passed, 7 skipped, 21 warnings in 47.96s
```

## Failures / errors

none

## Skip reasons (grouped)

```
  5  нет EXPDS_METRIC
  2  нет CUDA
TOTAL 7
```

## Notes

- GPU/CUDA / missing submission weights / optional deps → skipped via `koz3_box_env` plugin (not counted as regressions).
- Only existing dirs among `tests/`, `centerhead/tests` are passed to pytest (no exit-4 on missing centerhead).
- Official metric (`EXPDS_METRIC`) and real competition samples are not on this box.
- Full log: `/home/box/shared/koz3/box/results/bots-bot8-orsp-tta_35ca3df.log`

Re-run: `run_tests.full.sh bots/bot8-orsp-tta`
Cleanup worktree: `run_tests.full.sh --cleanup bots/bot8-orsp-tta`
