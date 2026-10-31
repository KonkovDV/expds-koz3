# koz3 test report: `bots/bot8-tta-lb` @ `f6fe20a`

- started: 2026-10-29 20:31:44 MSK
- finished: 2026-10-29 20:32:53 MSK
- resolved: `origin/bots/bot8-tta-lb` (`f6fe20a41b73e406c3a5ebd62134966e8c0bda6f`)
- worktree: `/workspace/koz3-wt/bots-bot8-tta-lb`
- venv: `/workspace/koz3-venv` (CPU-only torch)
- host: box «1» (8 CPU, low free RAM)
- command: pytest -q -rfEs -p no:cacheprovider -p koz3_box_env tests
- test paths: `tests` (only dirs that exist on this branch)
- wall time: 68.6s
- peak RSS: 1904.8 MiB
- exit: 0

## Counts

| passed | failed | errors | skipped |
|---:|---:|---:|---:|
| 387 | 0 | 0 | 7 |

## pytest summary lines

```
387 passed, 7 skipped, 21 warnings in 66.88s (0:01:06)
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
- Full log: `/home/box/shared/koz3/box/results/bots-bot8-tta-lb_f6fe20a.log`

Re-run: `/home/box/shared/koz3/box/run_tests.sh bots/bot8-tta-lb`
Cleanup worktree: `/home/box/shared/koz3/box/run_tests.sh --cleanup bots/bot8-tta-lb`
