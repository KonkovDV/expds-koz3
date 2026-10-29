# F:\expds3 audit (29.10.2026 ~09:55 МСК, bot-1) — read-only, nothing moved or deleted

F: free **12.35 GB** (used 1850.66 GB). Note: was 16.8 GB at 09:39 — something wrote ~4.4 GB to F: in 15 min (not ours; no process cmdline mentions expds3).
Train data is on **F:\2\train-expds-3** (not under F:\expds3). F:\expds3\train-expds-3 and F:\expds3\public_val do NOT exist; old scripts that point there are stale.

## 1. Top level
| path | GB | files | last mod | role | verdict |
|---|---|---|---|---|---|
| F:\expds3\work | 114.23 | 3206 | 10-28 19:04 | all OOF/stack/cache work (scoreboard `--work`) | see §2 |
| F:\expds3\mock_val | 11.28 | 139 | 10-22 | 48 mock regions — docker sim for v2/v3 (`run_v2b.cmd` uses mock_val\region_M047) | KEEP |
| F:\expds3\organizers | 0.07 | 74 | 10-25 | competition docs/materials, platform_logs, s3 sync | KEEP |
| eval_gnezdovo, eval_geo, mock_geo | 0 | 0 | — | empty dirs (eval_geo referenced by scripts\geo_rank.py) | KEEP (0 B) |
| _s3_listing.csv | ~0 | 1 | 10-20 | S3 listing | KEEP |

## 2. F:\expds3\work (dirs 95.1 GB + 511 loose files 19.1 GB)
### 2a. Must KEEP — feeds base A, accepted package, or current jobs
| path | GB | role | evidence |
|---|---|---|---|
| work\stack_arch.geojson | 1.05 | **base A stack** | = C:\Sloi\expds3c\in\stack_base.geojson (1 132 180 452 B, 10-28 00:02); PC_RESULTS 3c "База = F:stack_arch"; pkg\q3.cmd |
| work\stack_verify_sp_arch.geojson | 1.09 | crop layer of accepted bot-7 package | = expds3c\in copy (1 171 949 537 B); q3.cmd |
| work\gt_train_v2.geojson | 0.04 | GT for every honest-10 run | scoreboard default; copies in expds3c\in and box /workspace/rerank |
| work\fusion\ (li_hard2_fused.geojson …) | 3.49 | base lidar layer | scoreboard default `work/fusion/li_hard2_fused.geojson` |
| work\mosaic_li\ | 0.35 | base mosaic kurgans | scoreboard default `work/mosaic_li` |
| work\spor_gor_oof, spor_focal_oof, li_gor_mask_oof, geo_cur .geojson | small | base layers | scoreboard defaults |
| work\ae_fort_mask_oof, li_fort_mask_oof, gor_phys, sp_arch_oof, or_arch_oof (+ surf_arch_sp/or) .geojson | small | base-add layers of A (`ae_fort/li_fort/sp_focal2/gor_phys/sp_arch/or`) | EXPERIMENTS.md line 91; geo211\g211.cmd |
| work\li_hard2_oof\ (fold0–4, report.json, 10-21) | 2.29 | fold weights behind li_hard2_fused and the li_half dump lineage | work59\q2.cmd → lidump → pc_jobs\lihalf\work |
| work\cache\li_tin_59_t128 | 0.28 | lidar59 tiles cache (li_half/s5/lidar59 reruns) | REPORT2 commands `--cache li_tin_59_t128` |
| work\cache\ (other 40 dirs) | 10.99 | training/predict tile caches | 222 refs in scripts; regenerable but only by long F: reads → KEEP |
| work\li_fort_mask, ae_fort_mask, sp_arch_oof, surf_arch, spor_focal_oof, li_gor_mask, spor_gor_wide | 8.43+1.71+1.22+3.84+1.28+1.60+4.26 | fold weights behind base-add layers | conservatively KEEP (needed to regenerate/verify base layers) |

### 2b. UNCERTAIN — old experiment dirs (fold weights + OOF), mostly rejected lines; ask owner before TRASH
| path | GB | last mod | note |
|---|---|---|---|
| li_opn5_oof, li_hard3_oof, li_hard3_nocp_oof, li_cur3_oof, li_hard_oof, li_all_oof, li_tin_oof | 2.67+2.12+2.12+2.02+1.90+1.36+1.18 = 13.4 | 10-20…21 | early lidar variants; their .geojson outputs cited in notes/*.json |
| li_ema_oof, li_dev_oof, li_focal_oof, li_thin, li_gor_wide, li_gor_center | 2.02+2.01+1.58+2.15+1.60+1.57 = 10.9 | 10-21…24 | lidar variants; li_dev_far/asshipped referenced in notes |
| li_plus_oof | 2.49 | 10-28 08:12 | newest lidar line (li_tin_hard2_plus cache) — lean KEEP |
| spor_focal_r05, spor_focal_r34, spor_focal_photo, spor_focal_scale, spor_focal_hrrep, spor_focal_wide | 4.19+1.47+1.29+1.28+1.27+1.26 = 10.8 | 10-23 | sp_focal replacements (repl_sp_*.json); which one is `sp_focal2` in base is NOT confirmed → don't touch until resolved |
| spor_hardneg, spor_r1_oof, spor_all_oof, spor_center_oof, spor_large | 1.55+1.40+1.31+1.27+0.74 = 6.3 | 10-21…23 | spor variants |
| ae_focal_oof, ae_focal_r1, sp_arch_mask | 2.02+1.75+1.23 = 5.0 | 10-21…23 | optical variants |
| maggr_finds | 3.77 | 10-25 | maggr_finds_oof (geo finds; PIXGBM now replaces) |
| fold0…fold4 | 5×0.34 | 10-27 | unknown fold runs (10 files each) — inspect |
| loose: stack_base (10-26, old base), stack_rerank_first, stack_rerank_kurgans, stack_verify_sp, stack_arch_gate | 1.19+1.15+1.15+1.15+1.09 = 5.7 | 10-26…28 | superseded stacks; stack_base/stack_verify_sp still referenced by crop_verify.py defaults |
| loose: li_*_oof.geojson, alldata_dev*, ae_focal*_oof.geojson, tin_*.npy, ch_*.npz | ~8 | 10-20…24 | outputs of the dirs above |

### 2c. SAFE_TRASH candidates (small; smoke/test leftovers)
sub_unzip 0.27, archive_test 0.17, wheels 0.16, smoke_ctx 0.16, dist_test 0.15, smoke_in/out/out2/win/dock/coarse, probe_*, mock_out/pair/ship/hill/census/density/docker*, zipdiff, __pycache__, bench_in/bench_mix (empty) — total ≈ 1 GB. Some are referenced by analysis\cmp_smoke.py / make_smoke.py (dev tooling only).

## 3. References (C:\Sloi excl. TRASH/.venv/.git/docs; *.ps1 py json bat cmd yml sh <3 MB) — full list: PC C:\Users\Пользователь\f_refs_pc.txt (105 distinct paths)
- Active job configs: pkg\q3.cmd → stack_arch, stack_verify_sp_arch; work59\q2.cmd → li_hard2_oof; expds3c\bot7v2\run_v2b.cmd → mock_val\region_M047; geo211\g211.cmd → surf_arch_sp/or; wt_s4\scripts\s4_rerank.py → work\stack_s4_imnet.geojson (**not present on F:** — the only copy is C:\Sloi\work59); wt_bots\scripts → stack_subclass.geojson, kurgan_subclass_crops.npz, gor_coarse_dem.geojson.
- Bulk of refs (~95 of 105) come from expds-koz3_v0.2.0 analysis/scripts/notes (historical experiment records).
- Running processes at ~09:55: none with "expds3" in the command line.

## 4. Cross-check with current work
- v2 eb268d6 measure: reads C:\Sloi\expds3c\in copies (stack_arch, verify_sp_arch, gt_train_v2, pix_lofo) + mock_val via sim. → needs mock_val.
- v3 e2e: regions junctions → F:\2\train-expds-3 (101/103); full-mock_val sim needs F:\expds3\mock_val.
- li_half: pc_jobs\lihalf\work (own weights) + F:\2\train-expds-3. Lineage from work\li_hard2_oof.
- honest-10 base A: `--work F:\expds3\work` + base-add layers (§2a).
- stack_s4_imnet: not on F: → C:\Sloi\work59 must stay.
- stack_rerank_arch.geojson (accepted rerank R): not found among F:\expds3\work stack_* files — lives elsewhere (box /workspace/rerank) — verify before any cleanup.

## Summary
KEEP ≈ 53 GB (base/package/caches/mock_val), UNCERTAIN ≈ 68 GB (old experiment dirs + superseded stacks), SAFE_TRASH ≈ 1 GB. No action taken.
