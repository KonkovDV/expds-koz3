# C:\Sloi disk usage, measured 2026-10-29 09:39 MSK (read-only, nothing deleted)
Free: C: 0.5 GB of about 953 GB (critical). F: 16.8 GB.
C:\Sloi total is about 52 GB. The rest of C: (about 900 GB) is outside C:\Sloi.

| Entry | GB | Modified | Largest subfolders |
|---|---|---|---|
| expds3c | 12.16 | 10-29 | v2meas 5.01, regions 3.42, in 2.20, wt_bot7 0.80, bot7v2 0.49 |
| expds-koz3_v0.2.0 | 11.55 | 10-20 | .venv 4.82 (in use), dist 2.73, models 2.07, rust 1.19, .git 0.74 |
| oof59 | 6.96 | 10-28 | r18 3.43, r34 2.50, r34mask 1.03 |
| models_59 | 4.39 | 10-28 | li_r34_fpn_mask_59 0.67, raw_koz3-gpu-s1 0.67, -s2 0.60, li_r18_hard2_59 0.51 |
| pc_jobs | 3.26 | 10-29 | lihalf 2.15 (incl. 0.56 GB --out geojson), v3e2e 1.11 (v2 + v3on zips) |
| lidump | 2.89 | 10-28 | work 1.30, li_hard2_probs 1.07, li_hard2_oof_dump.geojson 0.52 |
| work59 | 2.63 | 10-29 | acc_s4 1.29, stack_s4_imnet 0.62, stack_base_59 0.60 |
| s5 | 1.81 | 10-28 | |
| sim | 1.29 | 10-29 | |
| wt_bots | 0.80 | 10-28 | |
| wt_lihalf | 0.80 | 10-29 | unused git worktree from me; patch did not apply |
| files | 0.41 | 10-27 | |
| kur | 0.34 | 10-28 | |
| everything else | <0.2 each | | about 30 small files (zips/md/pdf at about 0 GB), small wt_* worktrees |

Note: C: free space moves around. It was 0.5 GB at 09:39 and 4.66 GB at 09:42 MSK, after I removed about 0.2 GB of my own duplicate staging. Something outside C:\Sloi (temp/pagefile/other apps) is using several GB.
