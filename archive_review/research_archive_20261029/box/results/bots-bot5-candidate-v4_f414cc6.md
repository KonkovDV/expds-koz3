# koz3 test report: `bots/bot5-candidate-v4` @ `f414cc6`

- started: 2026-10-29 16:12:57 MSK
- finished: 2026-10-29 16:14:15 MSK
- resolved: `origin/bots/bot5-candidate-v4` (`f414cc697449b8980883ef6e938df5bc3d27b7bd`)
- worktree: `/workspace/koz3-wt/bots-bot5-candidate-v4`
- venv: `/workspace/koz3-venv` (CPU-only torch)
- host: box «1» (8 CPU, low free RAM)
- command: pytest -q -rs -p no:cacheprovider  (+ centerhead/tests separately)
- wall time: 51.5s
- peak RSS: 1747.7 MiB
- exit: 1

## Counts

| passed | failed | errors | skipped |
|---:|---:|---:|---:|
| 536 | 10 | 0 | 26 |

## pytest summary lines

```
```

## Failures / errors

none

## Skip reasons (grouped)

```
  9  нет вендора transformers: /workspace/koz3_shared/bot-7/work6/vendor_test/lib
  6  нет опорного входа смоука ORGATE (вне git, п. 13.11 ТР)
  5  нет EXPDS_METRIC
  2  нет CUDA
  1  нет артефактов бокса bot-5 (work/gd/root, analysis/gor_dense)
  1  нет трейна 008_Герите на машине
  1  нет опорного входа смоука ORGATE (вне git)
  1  нет распакованного вендора transformers 5.21.0
TOTAL 26
```

## Notes

- GPU/CUDA tests are skipped on this runner (CPU-only).
- Missing submission files (pixgbm / rerank weights / geo_pixgbm.py) cause intentional skips.
- Official metric (`EXPDS_METRIC`) and real competition samples are not on this box.
- Full log: `/home/box/shared/koz3/box/results/bots-bot5-candidate-v4_f414cc6.log`

Re-run: `run_tests.full.sh bots/bot5-candidate-v4`
Cleanup worktree: `run_tests.full.sh --cleanup bots/bot5-candidate-v4`
