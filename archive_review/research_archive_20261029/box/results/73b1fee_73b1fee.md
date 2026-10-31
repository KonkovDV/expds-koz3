# koz3 test report: `73b1fee` @ `73b1fee`

- started: 2026-10-29 20:03:46 MSK
- finished: 2026-10-29 20:05:01 MSK
- resolved: `73b1fee` (`73b1feed750e19252030c31af5ee2f363ae8fe64`)
- worktree: `/workspace/koz3-wt/73b1fee`
- venv: `/workspace/koz3-venv` (CPU-only torch)
- host: box «1» (8 CPU, low free RAM)
- command: pytest -q -rfEs -p no:cacheprovider -p koz3_box_env tests
- test paths: `tests` (only dirs that exist on this branch)
- wall time: 74.5s
- peak RSS: 1838.7 MiB
- exit: 0

## Counts

| passed | failed | errors | skipped |
|---:|---:|---:|---:|
| 424 | 0 | 0 | 7 |

## pytest summary lines

```
424 passed, 7 skipped, 21 warnings in 72.33s (0:01:12)
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
- Full log: `/home/box/shared/koz3/box/results/73b1fee_73b1fee.log`

Re-run: `box/run_tests.sh 73b1fee`
Cleanup worktree: `box/run_tests.sh --cleanup 73b1fee`
