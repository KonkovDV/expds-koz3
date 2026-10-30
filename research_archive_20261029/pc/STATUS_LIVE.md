# PC live status (Europe/Moscow)

Updated: 2026-10-29 ~20:00 МСК

## GPU lane
- **bot2 A3 @829ea7d: RELAUNCHED 17:30** (17:24 run FAILED: C: full during worktree add; wt now F:\Sloi_wt)
- bot8 honest-10 TTA/MS: waiting (after A3)
- bot5 dfine_aeor Or-only: expect after A3/bot8 window — **must use Docker**, not host venv
- bot3 B: after above

## CPU lane
- **bot6 PIXGBM @3f61ea8: RELAUNCHED 17:30 (wt F:\Sloi_wt)** 2 thr + `--match-reference` (Cyrillic-safe LGBM + REF params). wt `wt_bot6_3f61ea8`
- **bot5 gor heads @4dadeab: FAILED** (No module named transformers host venv) — **do not rerun**; bot5 will file Docker runner. schtask deleted.
- Thread budget: bot6=2; headroom for BACKLOG CPU filler

## Security
- Kaggle token must NOT live in shared koz3; use `$KAGGLE_API_TOKEN` env on box only. geo200 upload deprioritized.

## Prior DONE (unchanged)
- bot2 A2 fb66c89 DONE (ImportError n_raw=0) → A3
- bot4 LO 337a27c DONE FAIL Δ=−0.01629
- bot6 d173a75 DONE reproduce FAIL (superseded by 3f61ea8)
- bot7 diff aa3694f DONE Δ≈−0.020621
- bot3 aerial 475bafa DONE (sync to results still pending)


## lane stamp 29.10 17:39 MSK (CPU executor)
- CPU NOW: launching bot5 gor_heads Docker @9f62a96 THREADS=8 JobDir/Wt on F:
- GPU order: A3 @829ea7d → v2 base docker @10e9936 → bot8 → bot4 ablation @5097c4a (ALL gate e2e in pantum/n7q:20261024; no host venv gates)
- C free~13GB after archive v3e2e/lihalf/hotfix→F; new WT/out on F:


## lane stamp 29.10 17:46 MSK (CPU executor)
- bot7 compare_candidates @b6f835d **DONE** ENV=docker:pantum/n7q:20261024 → pc/results/bot7_compare_candidates_b6f835d/
- bot5 gor_heads Docker @9f62a96 **RUNNING** koz3-bot5-gor-heads-20261029174227 THREADS=8 (feature extract ~60+/173)
- GPU order still: A3 → v2 base docker → bot8 → bot4 ablation (all pantum/n7q)


## lane stamp 29.10 17:59 MSK (CPU executor)
- bot5 gor_heads Docker @9f62a96 **DONE** STATUS DONE; sizes 407736 / 204984; sha256 recorded; synced → pc/results/bot5_gor_heads_9f62a96/
- bot7 compare_candidates @b6f835d **DONE** earlier (ENV=docker); results + EXPERIMENTS appended
- GPU next (when free): A3 @829ea7d → v2 base docker → bot8 → bot4 ablation @5097c4a (all pantum/n7q; WT/out on F:)


## lane stamp 29.10 18:06 MSK (PC GPU+CPU executor)
- GPU NOW: **bot2 A3 @829ea7d RELAUNCHED** JobDir=F:\Sloi_pc_jobs\bot2_a3_dfine_docker_829ea7d Wt=F:\Sloi_wt\wt_bot2_829ea7d ENV=docker:pantum/n7q:20261024 EXPDS_STRICT=1 (prior 17:32 FAILED: zip path missing after v3e2e archive → empty zip_A / no main.py)
- GPU next: v2 base docker → bot5 gor_dense e2e @40fa599 → bot8 TTA/MS → bot4 ablation @5097c4a
- CPU: bot5 gor_heads @9f62a96 DONE; bot6 PIXGBM @3f61ea8 FAILED (pick_rasters arity) — not rerunning; bot7 compare DONE; headroom for BACKLOG filler
- Lease created: leases/pc_bot5_gor_dense_e2e_40fa599.lease QUEUED
- F free ~4.1GB after removing done wt_bot7_compare_candidates; kit/zips via _vendor_zips + archive


## lane stamp 29.10 18:08 MSK (PC executor)
- GPU: **A3 @829ea7d RUNNING** container koz3-bot2-a3-20261029180829 ENV=docker:pantum/n7q EXPDS_STRICT=1; fixed false R_drop_groups (comment match); zip_A already on F: (~736MB)
- GPU next: v2 base → bot5 gor_dense e2e @40fa599 (GPU, not CPU) → bot8 TTA/MS → bot4 ablation
- CPU: bot6 SKIP until new SHA; **bot8 compliance @35ca3df RUNNING** as BACKLOG filler (2 thr); F free ~3.4GB


## lane stamp 29.10 18:14 MSK
- GPU: A3 @829ea7d still RUNNING (docker koz3-bot2-a3-20261029180829)
- CPU: **bot6 PIXGBM @b0add08 LAUNCHED** KOZ3_OUT=C:\Sloi\pc_jobs\bot6_pixgbm_b0add08 thr=2 --match-reference
- bot5 (b') @40fa599 QUEUED GPU (not pick_rasters-affected); bot6 prior 3f61ea8 FAILED skipped
- F:~3.4GB C:~13GB — bot6 on C: by design


## lane stamp 29.10 18:21 MSK (PC executor)
- **GPU:** A3 @829ea7d RUNNING ~13m — SP_DFINE loading (DEPMETA warns only); region101 +43; container koz3-bot2-a3-20261029180829; F:~3.4GB
- **CPU (~10 thr target):**
  - **bot6 PIXGBM @b0add08 RUNNING** since 18:14 (already launched; thr=2, C:\Sloi\pc_jobs\bot6_pixgbm_b0add08) — extract ~214–215
  - **bot7 compare @b6f835d** relaunch after `pip install tzdata` (first fail ZoneInfo); JobDir/Wt on C:; archive paths `out_v3fix2`
  - bot3 aerial pytest: 44 pass / 1 fail WinError 1314 symlink (env) — FAILED note
  - bot4 toml/pytest filler RUNNING thr=2
  - bot8 compliance earlier: pytest 14 PASS; compliance FAIL features=0+cp1251
- **GPU next (queued):** v2 base docker → **bot5 gor_dense e2e @40fa599** (confirmed pick_rasters OK) → bot8 TTA → bot4 ablation
- Disk: F:~3.4GB (GPU only) C:~10.8GB (CPU outs) — never TRASH

## lane stamp 29.10 18:22 MSK
- CPU **bot7 compare DONE** (tzdata fix): v2=0.312509 PASS; hotfix_B/v4_off/bot4_off PASS; v3 FAIL sp_or; bot4_on FAIL li
- CPU **bot6 @b0add08 still RUNNING** thr=2 extract; C:~10GB
- GPU A3: sp_dfine model loaded cuda ~108s; still RUNNING
- bot5 (b') @40fa599 remains QUEUED for GPU after A3/v2


## lane stamp 29.10 18:26 MSK
- **A3 DONE_INFER** (docker_exit=-1 quirk): n_sp_dfine_in_state=**482**, n_raw=2506, pass_thr=13; ALL F1 **0.31053** (B 0.31287); sp_or 0.232 vs B 0.245 — D-FINE live but gate not improved
- **bot6 @b0add08 DONE_CHECK_VERDICT FAIL**: match-reference sha/metric mismatch (Δp 0.45/0.57); models on C:\Sloi\pc_jobs\bot6_pixgbm_b0add08
- **bot7 compare DONE** (earlier): v2 PASS; v3 FAIL sp_or; hotfix_B/v4_off/bot4_off PASS; bot4_on FAIL li
- **GPU next: v2 base docker LAUNCHING** JobDir=C:\Sloi\pc_jobs\bot2_v2_base_docker_eb268d6; then bot5 gor_dense @40fa599
- F:~3.4GB C:~10GB


## lane stamp 29.10 18:35 MSK
- **GPU NOW: v2 base docker RUNNING** koz3-bot2-v2base-20261029183101 @eb268d6 JobDir=C:\Sloi\pc_jobs\bot2_v2_base_docker_eb268d6 (~200s+); ENV=docker:pantum/n7q:20261024
- **GPU next:** bot5 gor_dense e2e @40fa599 (QUEUED) → bot8 TTA → bot4 ablation
- **A3 DONE_INFER+SCORED:** ALL F1 0.31053 (B 0.31287); locdump: n_pred_spor=57 min_same=0.70m median=2.02m within_2m=23/57 — localization OK for many; gate still worse on sp_or
- **bot6 @b0add08 already DONE_CHECK_VERDICT FAIL** (launched 18:14; sha/metric mismatch) — not relaunched
- **CPU fillers (~12 thr target, outs on C:\Sloi\pc_jobs):**
  - bot7 scoreboard recheck DONE (global+layer exit 0)
  - bot2 A3 locdump DONE
  - bot3/bot4/bot7/bot8 full pytest RUNNING (2 thr each)
  - bot8 compliance pytest 14 PASS (utf-8 re-run)
- Disk: F:~3.40GB C:~10.3GB — fillers on C:; never TRASH


## lane stamp 29.10 18:37 MSK
- **A3 DONE_INFER+SCORED** (container gone): n_raw 101=1497 / 103=1009 total=2506; in_state=482; pass_thr=13; ALL F1 **0.31053** (B 0.31287); sp_or 0.232 vs 0.245; synced pc/results/bot2_a3_docker_829ea7d/
- **GPU NOW: v2 base docker RUNNING** koz3-bot2-v2base-20261029183101 zip=**eb268d6** (QUEUE canon; STATUS older @10e9936 was tip typo) JobDir=C:\Sloi\pc_jobs\bot2_v2_base_docker_eb268d6 ~320s+
- **GPU next:** bot5 gor_dense @40fa599 → bot8 TTA → bot4 ablation
- **bot6 @b0add08:** DONE FAIL reproduction — **do not rerun**
- **CPU:** pytest fillers mostly DONE; scoreboard recheck DONE (g=0.297118 l=0.276497 Δ=-0.02062); locdump DONE; refilling lean CPU next
- Disk: F:~3.40GB C:~7.7GB — outs on C:\Sloi\pc_jobs; never TRASH


## lane stamp 29.10 18:39 MSK
- **GPU NOW:** v2 base docker **RUNNING** koz3-bot2-v2base-20261029183101 zip=eb268d6 (~7m+); JobDir=C:\Sloi\pc_jobs\bot2_v2_base_docker_eb268d6
- **GPU order next:** bot5 gor_dense e2e @40fa599 → **bot6 PIXGBM e2e-metric REF vs NEW @b0add08** (QUEUED lease; runner run_bot6_pixgbm_e2e_metric_b0add08.ps1; JobDir C:\Sloi\pc_jobs\bot6_pixgbm_e2e_metric_b0add08) → bot8 TTA → bot4 ablation
- **A3:** DONE_INFER+SCORED synced pc/results/bot2_a3_docker_829ea7d/ (n_raw 101=1497/103=1009; ALL F1 0.31053)
- **bot6 train @b0add08:** DONE FAIL reproduction — do not rerun; models at C:\Sloi\pc_jobs\bot6_pixgbm_b0add08\models\all ready for e2e-metric
- **CPU:** score_batch + bot2/bot6/bot7 pytest fillers RUNNING (~12 thr); outs on C:
- Disk F:~3.4GB C:~8.8GB — never TRASH

## lane stamp 29.10 18:45 MSK (GPU executor)
- **bot2 v2 base docker @eb268d6: DONE SCORED** ALL=0.312509 li=0.336983 sp_or=0.242915 ae=0.095556 Δ=0 vs kit; wall=680s; ENV=docker:pantum/n7q:20261024 → pc/results/bot2_v2_base_docker_eb268d6/
- GPU NEXT: **bot5 gor_dense e2e @40fa599** then **bot6 PIXGBM e2e-metric @b0add08** then bot8
- bot6 cleanroom train @b0add08 DONE FAIL — do not rerun; models reused for e2e-metric
- F:~3.4GB C:~8.7GB; bot5 Wt→C:; Heads from F:\Sloi_pc_jobs\bot5_gor_heads_9f62a96; KitGt=out\gt.geojson (out_v3fill2 empty of gt)

## lane stamp 29.10 18:52 MSK (GPU executor)
- v2 base docker @eb268d6 DONE SCORED ALL=0.312509 exact kit (gate baseline)
- GPU order: **(b') gor_dense @40fa599** -> **(c') dfine_aeor Or-only @2b745df** -> bot6 PIXGBM e2e-metric @b0add08 -> bot8 TTA -> bot4 ablation
- (c)@999572d SUPERSEDED by (c')@2b745df; lease pc_bot5_dfine_aeor_or_2b745df QUEUED; JobDir C:\Sloi\pc_jobs\...
- scorer fix: tools_v3e2e.tar lacks e2e_score.py -> copy from F:\...\out\tools for subsequent runners
- bot5 (b') relaunch: ASCII runner (prior parse fail from Cyrillic/emdash)

## lane stamp 29.10 18:54 MSK (GPU executor)
- GPU order NOW: **(b') gor_dense @40fa599 RUNNING** -> (c') dfine_aeor Or @2b745df -> **A4 agreement_m=50 @c3d912c** -> bot6 PIXGBM e2e-metric @b0add08 -> bot8 TTA -> bot4 ablation
- A4 lease pc_bot2_a4_c3d912c QUEUED; JobDir C:\Sloi\pc_jobs\bot2_a4_dfine_agreement_c3d912c; runner patched C:+scorer+salvage
- v2 docker baseline ALL=0.312509 (exact kit); gate vs host B 0.31287 for A4

## lane stamp 29.10 19:01 MSK (GPU executor)
- **PC OFFLINE since ~18:58 MSK** — machineId 08e3443c spawn/unreachable; likely sleep/off. Retrying every few minutes.
- Last known: (b') gor_dense @40fa599 was relaunched 18:59 after FAILED missing pixgbm (filled from A3 zip); STATUS RUNNING worktree HEAD ok; docker not yet observed before disconnect.
- GPU order when back: **(b') @40fa599 resume/relaunch** -> (c') dfine_aeor Or @2b745df -> A4 agreement_m=50 @c3d912c -> bot6 PIXGBM e2e-metric @b0add08 -> bot8 TTA -> bot4 ablation
- Or weights staged on box: /workspace/kaggle/pulls/koz3-dfine-or + chunks /workspace/dfine_or_chunks; thresholds at pc/weights/dfine_or/thresholds_or_oof_provisional.json (need CopyFromBox when PC back)
- v2 docker baseline ALL=0.312509 exact kit; C: was ~29GB free F:~3.5GB
- Disk policy: JobDirs on C:\Sloi\pc_jobs; never TRASH

## lane stamp 29.10 19:08 MSK (GPU executor)
- **PC OFFLINE again** (desktop app disconnected after brief reconnect ~19:04). Retrying with backoff.
- **URGENT (b') diagnosis DONE while briefly online:**
  - Container koz3-bot5-gordense-off-20261029190003 was RUNNING but INVALID: health showed `li_mask:li_gor_mask_r18.pt: 0.0` → torch.load EOFError Ran out of input in li_mask_feats.
  - Cause: wt file `C:\Sloi\pc_jobs\wt_bot5_gor_dense_e2e\models\li_gor_mask_r18.pt` was **0 bytes** (sha empty); robocopy /XC /XN /XO skipped refill because empty stub existed.
  - **STOPPED** container + killed runner pid 21328/6344. Restored from ModelsFill (`C:\Sloi\wt_bot5_50f6f27\models\li_gor_mask_r18.pt` 57426355 bytes sha39702CD8…).
  - **v2/A3 scores OK**: both zip mounts have good 57426355-byte copy (same sha). Only bot5 wt was empty. (Note: `bot2_a3_dfine_docker_import\zip_A\...` has truncated 1.7MB copy — unused by scored A3 which used F:\…\zip_A).
  - Cleared invalid out_off; patched runner: drop 0-byte stubs + robocopy /IS /IT + preflight size>0 + host torch.load smoke on critical .pt. Same empty-weight preflight added to A4/(c')/bot6 e2e runners. Patched file in Downloads\_runners — **NOT YET installed to JobDir/relaunched** (PC dropped).
- **When PC back:** (1) Copy Downloads\_runners\run_bot5… → JobDir\run.ps1, confirm mask size>0, relaunch (b'); (2) start CPU fillers ≤12 thr: bot7 compare_candidates @b6f835d (use docker v2 baseline geojson), pytest latest bot sha, cpu_score_batch; (3) continue GPU queue (c')→A4→bot6→bot8→bot4 after (b') DONE.
- Or weights still on box (/workspace/kaggle/pulls/koz3-dfine-or + chunks); thresholds at pc/weights/dfine_or/.
- Disk policy: JobDirs C:\Sloi\pc_jobs; never delete TRASH.

## lane stamp 29.10 19:17 MSK (GPU executor)
- PC ListMachines connected=true; **CopyFromBox OK**; **Shell spawn EPERM** since ~19:08 (retrying backoff).
- Staged on PC Downloads\_runners: run_bot5_gor_dense_e2e_docker.ps1 (preflight), pc_bootstrap_19xx.ps1 (install+relaunch bot5 + CPU fillers), run_bot7_compare_candidates_b6f835d.ps1
- Blocked: install patched runner → JobDir, relaunch (b'), start CPU. Mask already restored to 57MB on disk before disconnect.
- Action when Shell returns: `powershell -File Downloads\_runners\pc_bootstrap_19xx.ps1`

## lane stamp 29.10 19:23 MSK (GPU executor)
- **(b') @40fa599 RUNNING healthy** koz3-bot5-gordense-off-20261029192029; mask li_gor_mask_r18.pt **54.8 MB** (was 0.0 — restored); no EOFError in new logs.
- bot1 ran bootstrap/relaunch; executor did **not** re-run bootstrap. Verified mask+container; left GPU job alone.
- **CPU fillers started** (≤12 thr): cpu_score_batch (vs docker v2 baseline), bot7_compare_candidates_v2docker (V2=out_mock), cpu_pytest_bot5_tip, cpu_pytest_bot8_compliance.
- Preflight runners staged: JobDir (c')/(A4) run.ps1 + Downloads\_runners (empty-weight gate). bot5 run.ps1.next ready after current job.
- GPU next after (b') DONE: **(c') @2b745df** (need Or weights CopyFromBox) → A4 @c3d912c → bot6 e2e-metric → bot8 → bot4.
- v2/A3 scores unaffected (their zip mounts had good 57MB mask).

## lane stamp 29.10 19:36 MSK (GPU executor)
- **(b') gor_dense @40fa599 DONE** OFF+ON docker pantum/n7q; schtask koz3_bot5_gordense; container …-off-20261029192239 then ON …-on-20261029192419.
- Scores ALL/li/sp_or/ae = **0.312873 / 0.336983 / 0.245064 / 0.095556**; ΔOFF→ON = **0.0** (n=246 both; gor_dense added 4 pts then NMS). Same ALL as v4/hotfix_B baseline.
- Preflight: li_gor_mask 54.8MB; torch_preflight OK n=17; empty-weight gate + torch_preflight.py file fix (PS5 -c quote strip).
- Lean sync → pc/results/bot5_gor_dense_e2e_40fa599/ (scores/STATUS/warn_scan/heartbeat).
- **CPU:** cpu_score_batch **DONE** (schtask koz3_cpu_score_batch; fixed $D→$JobDir path bug). Summary: v2_docker=0.312509; A3=0.310530; hotfix_B/v4_off/gordense_on/bot4_off=0.312873; bot4_on=0.296584.
- pytest bot5 tip: DONE FAIL exit=1 (43 pass / 1 fail missing pixgbm in lean wt) via schtask koz3_cpu_pytest_bot5_tip.
- compare_candidates: FAILED exit=2 (`--out-dir` vs required `--out`) — patch pending; do not session-launch.
- GPU next: **(c') dfine_aeor Or-only @2b745df** — Or weights CopyFromBox in progress (20MB chunks → Downloads\_dfine_or; assemble → C:\Sloi\pc_weights\dfine_or). Runners A4/bot6/(c') already have torch_preflight.py + empty_weights + ASCII. Launch **only via schtasks**.
- Do NOT relaunch (b'). bot1 19:20 container stopped on purpose.
- Disk: JobDirs C:\Sloi\pc_jobs; never TRASH.

## lane stamp 29.10 20:00 MSK (GPU executor)
- **GPU NOW:** bot4 lo ablation (orchestrator launched ~19:52; 7 variants; ETA ~20:30). Do NOT start another GPU job until it finishes.
- **GPU next after bot4:** A4 @c3d912c → (c') Or @2b745df → bot6 r2 @b0add08 → bot3 f4 → bot8 TTA → bot3 final.
- **(b') gordense @40fa599 DONE** ALL/li/sp_or/ae=0.312873/0.336983/0.245064/0.095556 Δ=0; lean sync pc/results/bot5_gor_dense_e2e_40fa599/.
- **v2 PIXGBM:** LOADED backend=lightgbm (main_mock.log geo_pixgbm lines) → pc/results/bot2_v2_base_docker_eb268d6/pixgbm_check.txt.
- **bot6 e2e r1 INVALID** (geo_pixgbm.py missing in wt → classic fallback; gate parser false FAIL on delta=0). Queue r2 runner run_bot6_pixgbm_e2e_metric_b0add08_r2.ps1 after (c').
- **Or weights:** CopyFromBox 20MB chunks in progress → Downloads\_dfine_or (f0 parts 00-03 done; need 04-05 + f1/f2); assemble → C:\Sloi\pc_weights\dfine_or with sha check before (c').
- **bot3 f4 weights:** chunks ready on box /workspace/dfine_ae_ft_f4_chunks20 (sha249b86fb…); copying while GPU busy.
- **CPU:** bot7 compare DONE; cpu_score_batch DONE; pytest tip prior FAIL (1 miss pixgbm). Next: bot8 compliance schtask + bot5 pytest @ed73c5e clean wt. Stale schtasks with 23:5x Next Run DISABLED.
- Launchers only via schtasks; ASCII; torch_preflight.py file (no -c).
