# bot7 compare_candidates @ b6f835d (2026-10-29 17:45 Europe/Moscow)

Scoring ENV=`docker:pantum/n7q:20261024` (container `koz3-bot7-compare-20261029174540`).
GT=`F:\Sloi_pc_jobs_archive\v3e2e\out_v3fix2\gt.geojson` sha256=`e0e72411bd79f44b0784b34aa41ff9b7a2700c420236d25077b21a0126827487` (archived from C:\Sloi\pc_jobs\v3e2e; same honest-10 family as prior gates, n_gt ALL=3620 in bot5/bot7 notes).

Baseline ALL=0.312509 li=0.336983 sp_or=0.242915 ae=0.095556

| candidate | inference_env | ALL | li | sp_or | ae | dALL | rule |
|---|---|---:|---:|---:|---:|---:|---|
| v2 | docker (v3e2e) | 0.312509 | 0.336983 | 0.242915 | 0.095556 | 0.000000 | PASS |
| v3 | docker (v3e2e) | 0.287985 | 0.340909 | 0.000000 | 0.095556 | -0.024524 | FAIL(sp_or) |
| hotfix_B | host | 0.312873 | 0.336983 | 0.245064 | 0.095556 | 0.000364 | PASS |
| v4_off | host | 0.312873 | 0.336983 | 0.245064 | 0.095556 | 0.000364 | PASS |
| bot4_off | host | 0.312873 | 0.336983 | 0.245064 | 0.095556 | 0.000364 | PASS |
| bot4_on | host | 0.296584 | 0.321598 | 0.245064 | 0.095556 | -0.015925 | FAIL(li) |

Notes:
- hotfix_B / v4_off / bot4_off share identical geojson sha256 `fb11f3d9…` (flags-off identity).
- v4_off geojson path: `C:\Sloi\pc_jobs\bot5_v4_off_50f6f27\out\result.geojson`
- li↔ae kurgany cross-TP = 32 for all rows.
- Inputs archived under `F:\Sloi_pc_jobs_archive\` after C: reclaim (v3e2e, hotfix, bot4).
