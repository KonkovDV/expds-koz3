# DISK AUDIT 02.11.2026 (~16:10 MSK, UTC+3) — executor for A (Elena t292u)
Disk `/` 128G: **before 117G used / 11.7 GB free (92%)** → **after 90G used / 40.7 GB free (71%)**. Freed **≈29 GB**.
Logs: `DELETE_LOG_20261102.txt` (each path + size), `sha_proof_20261102.txt` (checksums of the duplicates before deleting).
Kaggle at audit time: elenakonkova/koz3-w1-satdet-frcnn RUNNING · dmitrijkv/koz3-w2-dfine2-b RUNNING · **dmitrijkv/koz3-w2-dfine2-a COMPLETE**.
No processes had cwd/open files under any deleted path. No git push, no LB, no secrets touched.

## DELETED (proof)
| path | size | why safe |
|---|---|---|
| v4build/kpar/ds_v4/v4c1f.zip.bin | 690M | sha f0def982… == w4geo/dist zip; Kaggle DS aaaa0000004/koz3-w4-v4c1f ready; v4chk PASS. Small files in ds_v4 kept |
| /tmp/fr1h-gh-upload/submission_fr1h_on_eb268d6.zip | 501M | sha fe8d60b9… == out/fr1h-pack zip (kept). Small files kept |
| out/fr1h-pack/drive-parts/*.part0-5 | 501M | cat(parts) sha fe8d60b9 == local zip (kept); Drive folder 17N5Cu… has same parts (sizes verified via Drive API) |
| out/fr1h-mrp-pack/drive-parts/*.part0-5 | 501M | cat(parts) sha e5b6c3b8 == local zip (kept); Drive parts verified |
| out/geo-clean-pack/pixgbm-parts/*.part00-06 | 552M | cat(parts) sha ed51cae4 == local zip (kept); Drive parts verified |
| out/sat2-f0/{grid,pack,ds_dmitrijkv} | 9.9G | sat2 closed (G1 sat2-kp FAIL, v3_on+sat2 FAIL — FINAL_CANDIDATES); training caches already uploaded as Kaggle DS sat2-f0-data/f14-data (both COMPLETE); rebuild via orch/bot-W1/sat2_code/build_grid_cache.sh; not referenced by satdet/dfine2/W4 gate scripts. Kept: kout, GATE_*.json, arm_*/fused_*.geojson, replace/, md. soft-PASS replace pack on Drive (sat2_det2_oof_replace_pack.zip) |
| wt/peak-rerank, wt/within-site-rank, wt/dino-gor, wt/ring-honest, wt/status-r50c5-reject, wt/status-fr1h | 5.3G | CLOSED STOP per HANDOFF_NOW_20261101 (peak-rerank, WSR, ring, dino-gor) / status-only PR branches; all clean (0 modified/untracked), no branch lease; removed with `git worktree remove` (no --force) → branches/commits stay in expds-koz3 (bots/a-peak-rerank, a-within-site-rank, a-dino-gor, status-fr1h-pass, status-r50c5-reject; d635a6b) |
| orch/bot-W1/e2e_v7/submission_v7.zip + submission_base_v4.zip | 1.24G | V7 D-FINE REJECTED (STATE); Δ recorded in REPORT_DELTA.md; sha in SHA256SUMS.zips (verified OK before rm); rebuildable from e2e_v7/tree + V7_local.patch |
| /tmp/b8smoke_{094ddsow,mqsgdmyc,1soaiias} | 2.0G | smoke-extraction temp dirs (01.11, v4 build smoke 02.11 14:55); v4 smoke+parity PASS done; no refs |
| /tmp/v3a, /tmp/v3am | 1.1G | extracted v3a/v3a_min; models already copied to backups/v3a_min_pix_clean_models_from_tmp (SHA256SUMS) per PREP_PIXGBM_CLEANROOM |
| /tmp/s3v, /tmp/fpenv, /tmp/augdeps | 1.95G | throwaway tmp venvs/deps; no script refs |
| /tmp/{splarge,wsr}_smoke.pt, /tmp/sat2in, /tmp/sat2out | 0.33G | smoke of closed sp_large/WSR/sat2 |
| /home/box/.cache/pip | 0.7G | pip download cache |

## DEDUPED (hardlink, path still valid, bytes identical)
- out/fr1h-gate/fr1h/work/geom_wkb.pkl → w5data/work/geom_wkb.pkl (446M, sha 552101ef both)
- e2e_v7/mock_val_n/* → identical files in e2e_v7/train_n (≈1.8G incl. 1.5G Миловидово Ae tif); cmp-verified
(fr1h-gate table.pkl ≠ w5data table.pkl → both kept)

## KEPT (large) — why
| path | size | reason |
|---|---|---|
| w4geo/dist (v4 zip + v4_drive_parts) | 1.4G | MUST KEEP final |
| v4build/res, v4build/gate | 1.6G+0.16G | **live gate inputs**: orch/bot-W4/comp scripts read v4build/gate/{v3_gate_lib.py,out/prep.pkl,x_c1fx} and v4build/res/orch/bot-W1/work/v3prod/t_taus.json |
| v4build/in/koz3_v3on_b447649…zip.bin | 690M | only local copy of rollback v3_on (1406c01a); Drive has it but kept local for fast rollback |
| v4build/v3on_x | 762M | extracted v3_on (REF for parity/C1F diffs) — UNSURE |
| w4geo/work | 22G | clean-room pixgbm workspace: E1_G1 FROZEN candidate (cstar), geo5 multi-seed models, feat caches listed as inventory in PREP_PIXGBM_CLEANROOM; not clearly closed → KEEP/UNSURE (biggest remaining item; ask Elena) |
| w5data/sat_raw | 5.2G | raw SpOr tifs; not referenced by satdet/dfine2 code, but satdet track is live (RUNNING) — UNSURE, re-downloadable from S3 |
| w5data/work, stack_arch.geojson, gt_train_v2 | 2.0G | honest-10 / satdet t1_prep inputs |
| w4geo/{spor_oof,eval,comp,train,ref} | ~3.1G | gate inputs / canon train data |
| orch/bot-W1/e2e_v7 (tree, train_n, mock_val_n) | ~3.6G | registered worktree + docker platform_sim mock; reusable for e2e — UNSURE |
| orch/bot-W3/kaggle_geo5 / gordense_elena_ds / seeds_b_out | 2.8G/0.8G/0.2G | W3 Kaggle staging/outputs; gordense heads unique (legacy not found) — KEEP |
| /tmp/gord_ds | 0.8G | gor_dense/v3 head npz (unique pair per GORDENSE_FIX_STATUS) — KEEP |
| w4geo/kaggle ds_v2clean/v2clean.zip.bin | 542M | local copy of fallback v2clean |
| w4geo/kaggle ds-v3a-min-check/v3a_min.zip.bin | 542M | superseded v3a_min, not found on Drive → UNSURE |
| out/{fr1h-pack,fr1h-mrp-pack,verify-spor-pack,geo-clean-pack} zips+tree | ~3.4G | one local copy of each pack (verify-spor NOT on Drive) |
| out/li_aug, out/r50c5-*, out/sat2-f14, out/fr1h-gate | 1.8G/1.2G/0.5G/0.8G | closed lines but contain OOF/weights/gate evidence; not on Drive → UNSURE |
| wt/{orch,w1,w3,w4,w5,coord-comp,comp-gaps,w4-pix} | ~7G | active worktrees (coord-comp has satdet tools) |
| wt/sp-large-f0, wt/e1g1-backup, wt/lidar-aug | 2.7G | DIRTY (uncommitted changes) and/or e1g1 not on remote; lidar-aug has active branch lease → not removed |
| wt/geo-clean | 823M | branch not on remote — kept |
| venv_sat2, /workspace/tvenv, venv-sanek, venv-fsp, w4geo/venv, venv | ~4G | tvenv/venv_sat2 touched today 12:13; venv-sanek/fsp belong to other projects (fsp-talent) |
| expds-koz3/.git | 931M | repo |
| ~/.cache/huggingface | 437M | model weights cache |

## BLOCKERS / NOTES
- satdet gate scripts (wt/coord-comp/tools/satdet/run_det.sh, satq) point at **/workspace/koz3/w1/**, **/workspace/koz3/w4venv**, orch/bot-W4/work/v3_gate_lib.py — these paths **do not exist on this box** (equivalents live in v4build/gate + v4build/res). Needs repointing before satdet gate runs.
- expds-koz3/.git/objects/pack/tmp_pack_9Gtpta (132M garbage) left alone (could be another agent's in-flight fetch).
