# koz3 test report: `bots/bot5-candidate-v4` @ `40fa599`

- started: 2026-10-29 18:06:10 MSK
- finished: 2026-10-29 18:07:18 MSK
- resolved: `origin/bots/bot5-candidate-v4` (`40fa5997a046c8b08fbb65af0a5f043dd0fd75af`)
- worktree: `/workspace/koz3-wt/bots-bot5-candidate-v4`
- venv: `/workspace/koz3-venv` (CPU-only torch)
- host: box «1» (8 CPU, low free RAM)
- command: pytest -q -rfEs -p no:cacheprovider -p koz3_box_env tests
- test paths: `tests` (only dirs that exist on this branch)
- wall time: 67.2s
- peak RSS: 1972.0 MiB
- exit: 0

## Counts

| passed | failed | errors | skipped |
|---:|---:|---:|---:|
| 541 | 0 | 0 | 37 |

## pytest summary lines

```
541 passed, 37 skipped, 22 warnings in 65.29s (0:01:05)
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
- Full log: `/home/box/shared/koz3/box/results/bots-bot5-candidate-v4_40fa599.log`

Re-run: `box/run_tests.sh bots/bot5-candidate-v4`
Cleanup worktree: `box/run_tests.sh --cleanup bots/bot5-candidate-v4`
