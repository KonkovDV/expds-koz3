# TRASH_RECHECK (29.10.2026 ~09:50 МСК, bot-1) — nothing deleted

Source of evidence: PC listing of C:\Sloi\TRASH (09:45), Select-String over C:\Sloi (excl. TRASH, .venv, .git, site-packages; *.ps1 py md json bat cmd yml yaml toml txt sh log <5 MB), Win32_Process command lines (09:46), bot-9/PC_RESULTS.md, box copies. PC went OFFLINE ~09:47, so sha256 comparisons PC vs box/F: were NOT done.

## Verdicts
| folder | GB | verdict | key reason |
|---|---|---|---|
| oof59 | 6.96 | SAFE (bulk) | OOF of r18/r34 lidar59 — both REJECTED (r18 −0.025; r34 FPN mask Δ −0.0076). Refs only in historical wt_lidump/analysis/2026-10-28/oof59_pc scripts. Keep the small sb_*.json/diag_*.json (evidence) |
| models_59 | 4.39 | UNCERTAIN | weights of rejected models, but PC is the ONLY full copy: box has only li_r18_hard2_59_fold0/1.pt (kaggle_out/koz3-li-tin-59-t128-dual-t4); no final.pt, no r34 edge/mask, no raw_koz3-gpu-s1/s2 on box. Re-download from Kaggle possible only while outputs live |
| lidump | 2.89 | SAFE (likely) | box has bot-2/li_dump/li_hard2_oof_dump.geojson (555 240 061 B), li_hard2_probs/, lidump.log; fold weights copied to pc_jobs\lihalf\work (li_half used those, not lidump). sha not re-checked |
| work59 | 2.63 | KEEP — move back | PC_RESULTS: s4 rerank r18_imnet ACCEPTED (Δ +0.00729), "Файлы ПК: C:\Sloi\work59\stack_s4_imnet.geojson, sb_s4_imnet.json, acc_s4\accept_s4_imnet.json" — only known location. Live scripts point to work59 |
| s5 | 1.81 | SAFE | cfix OOF REJECTED (r18 Δ −0.0081, r34 Δ −0.0030); outputs regenerable from Kaggle s5 tar. Keep sb_cfix_*.json |

## Path references found outside TRASH (point 2)
- work59: bench\run_bench.ps1:15 (bench_done.flag); geo\geo.cmd:15 and sim\chain2.cmd:3 wait for work59\pix_done.flag; pix\pixmain.ps1:6,36 (tail_done/pix_done flags); pkg\q3.cmd:11 `set G=C:\Sloi\work59\gt_train_v2.geojson`, :14 work59\q2.log; xfer\run_seq.cmd:6,8 (seq.log). All these queues are finished (last outputs 00:10–00:41), but re-running any of them would hang or fail without work59.
- oof59: only wt_lidump\analysis\2026-10-28\oof59_pc\{fuse_r18.py,pipe_r18.cmd,push.ps1,results_r18.txt} and README (historical analysis).
- models_59: only `/workspace/expds_work/models_59` (box path) in scripts\kaggle_collector_daemon.py and the REASONING log in every worktree — a box path, not C:\Sloi.
- lidump, s5: no references in C:\Sloi configs.
- Processes at 09:46: none reference these folders (only Docker Desktop, an unrelated Claude cmd.exe, and our powershell). No waiting geo/chain2/pix loops.

## Current candidates (point 3)
- expds3c bot7v2 (eb268d6): models are hard links in expds3c\wt_bot7\models; base stack_arch on F:; no refs to TRASH folders.
- v3 e2e (pc_jobs\v3e2e): kit\ zips + regions junctions to F:\2\train-expds-3; no refs.
- li_half: pc_jobs\lihalf\work (own copy of fold weights); no refs.
- A (bot-10 REPORT4) uses rerank_arch, not stack_s4_imnet; but s4_imnet is an accepted candidate that is still waiting to be combined, so its files must stay.

## Copies elsewhere (point 4)
- gt_train_v2.geojson: box /workspace/rerank/gt_train_v2.geojson 39 401 752 B (≈ PC 37.6 MiB) — duplicate, but q3.cmd expects it at work59.
- li_hard2_oof_dump.geojson + probs: on box (bot-2/li_dump).
- li_r18_hard2_59 fold0/1: on box; fold2–4, final, all r34: PC only.
- stack_s4_imnet / acc_s4: whether a copy exists at F:\expds3\work\stack_s4_imnet.geojson (the REPORT2 command path) is NOT verified (PC offline).

## Recommendation
1. Move work59 back: `Move-Item C:\Sloi\TRASH\work59 C:\Sloi\work59` (same volume, instant).
2. Keep models_59 in TRASH (not deleted) until bot-9/bot-10 confirm the r18/r34 lidar59 models are dead, or until fold2–4/final are copied off the PC.
3. Deleting oof59, s5, lidump would free ~11.7 GB on C: — that needs the user's explicit go; keep the small sb_*/diag_* json first.

## Update 09:53 MSK
- work59 restored to C:\Sloi\work59 (stack_s4_imnet.geojson present; F:\expds3\work has NO stack_s4_imnet => work59 is the only copy).
- models_59 restored to C:\Sloi\models_59 by user order (113 files, 4.39 GB).
- TRASH now: lidump, oof59, s5 (nothing deleted).
