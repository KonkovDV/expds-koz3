# EN_COMMIT_GAP_VS_US_20261101 — EN score-up steps vs our tree, feature by feature

Author: A-executor (research subagent), 01.11 ~11:55 MSK. Tasks t228u (priority) + t225u/t227u FULL GO.
Sources (read-only, blobless clone `out/competitors/сторонний репозиторий_EXpedition`, branch `origin/codex/v24-dinov3-1024`):
`docs/EXPERIMENTS.md` (version ledger with public scores), `docs/WEAKNESSES.md`, `docs/PAPER_TRIAGE_AND_P2_2026-10-29.md`,
`docs/*_VERDICT_*.md`, `git log --all` (94 commits, 26–29.10). **Only docs and commit messages were read. No EN .py / weights
were copied, opened into, or referenced from our tree.** Our side: `repo/` @70dedf4 (+ deployed config `configs/default.toml`),
`repo/kaggle/2026-10-28/convnext_lidar/src/convnext_main.py` (our cnx trainer), `orch/bot-W2/PREREG_W2_cnx.md`, `orch/OFFICIAL_SCORES.md`.
LB 0.41238 (eb268d6) untouched; nothing submitted.

## 0. TL;DR (ranked gaps by expected LB Δ for us)

| Rank | Gap (EN has, we don't / only partly) | EN evidence | Our status | Exp. LB Δ for us | Close under FULL GO |
|---|---|---|---|---|---|
| G1 | **Heterogeneous satellite detector fusion** (4–6 members, mean score, absent=0, r 8 m, thr 0.25) | v9→v12: sat P/R .381/.599 → .442/.642, sat-nested 0.307 → 0.367; LB 0.417→0.489 jump is mostly this + framing/hedge | sp_focal + sp_focal2 (both R18 focal-centre family), sp_dfine **off** (unreproducible) | **+0.02…+0.05** (gap model: sat to EN level = +0.051) | SAT2 (EN_PLUS §5): 2nd/3rd *different* family + quality aug + site balance, OOF on same folds, mean-score fusion |
| G2 | **LiDAR physically-correct aug: relief recomputed after height/zoom of the DTM** | LI_AUG single 0.5327 → 0.5490 (li-nested); site recall Esbyu .35→.63 | our cnx scales **already-clipped u8 relief** (`convnext_main.py` L449–452) and bilinear-resizes relief (L417–433) → tall mounds stay saturated, LRM window not physical | +0.01…+0.02 | lidar-aug executor (PROMPT_LIDAR_AUG_SEEDS §B1): precomputed DTM-variant relief banks |
| G3 | **LiDAR site-balanced sampling** | +0.012–0.016 (WEAKNESSES #21) | our WinDS samples objects ∝ class weight only (L401–403) → Gnezdovo/Timerevo dominate | +0.005…+0.012 | PROMPT §B2 (site-first sampler, α=0.5 temperature) |
| G4 | **LiDAR openness± / SVF / copy-paste channels+aug (augx)** | augx single 0.5688 vs aug 0.5490 (+0.020) | c5 (svf_r, opp_r) + copy-paste exist but c5 line closed (bot-1), c3 used | +0.005…+0.015 | PROMPT §B3: 7-ch set incl. neg-openness + MSRM; c5 re-opened only with B1/B2 |
| G5 | **Recipe-diverse LiDAR blend** ((s0+s1)/2 + aug + augx, thirds) | blend 0.5765 vs best single 0.5688; seed-only 2nd augx no gain | cnx s012avg (3 seeds same recipe) **FAILED** stack gate (P-W2-6) | +0.005…+0.01 | PROMPT §B5: 3 *recipes* × 1 seed > 3 seeds × 1 recipe |
| G6 | **LiDAR as a standalone expert replacing li-kurgans**, not an additive layer | EN LiDAR expert = the U-Net blend only | our cnx was gated additive + dedupe 12 m vs li_r18_hard2; replacement (b) also failed for c3-sc | enabler for G2–G5 | PROMPT §C: primary gate = replace li kurgan groups (b-kur) **and** additive, decided on li-only + stack |
| G7 | **Neighbourhood prior inside the LiDAR decoder** (peak × (0.5+0.5·clip(n/K,0,2)), R 200, K 3, snb 0.2) | li-nested 0.5765 → 0.5831; platform LiDAR R .637 → .676 (+0.039) for −0.007 P | implemented `expds/post.py:901 kurgan_neighbour_prior`, **off**; on stack A li-groups +0.0002 P 0.63 → closed | +0.003…+0.008 **only on the new heatmap** | PROMPT §B6: apply to new cnx peaks before threshold; separate gate |
| G8 | Satellite d4-TTA | sat-nested 0.326 → 0.339 (7c70d5f) | not measured on spor_focal | +0.003…+0.01 | SAT2 step S0 (inference-only OOF, Kaggle GPU) |
| G9 | Satellite framing prior (d_rel>0.5 needs ≥0.35) | CPU pipeline 0.279 → 0.297; part of v12 | **FR1h ready**, gate +0.00409 P 0.999 | +0.004 | needs Elena «да» on LB (not ours to submit) |
| G10 | Hillfort localiser (DINOv3-SAT) | v23/24 sat-nested 0.3797 → 0.4097/0.4159 | **DINOv3 banned**; DINOv2-L (T8) not done | +0.005…+0.02 | later, only if DINOv2-L allowed |
| G11 | Ae fortifications (FCOS) | v20 +0.003 public | ae_fort_mask_r18 present | ≈0 | no action |
| G12 | EtGr layer isolation / one geo pass per region | v19b +0.0098 | **have** (ETGR_MAX_DEPTH_M, layer preserved) | 0 | — |
| G13 | Pixel-GBM geophysics | +0.003 (v19b) | **have** (eb268d6), clean-room P1 PASS | 0 | A1 budget only |

## 1. EN public score timeline (from `docs/EXPERIMENTS.md` headings) → what moved it → us

| EN version (date) | Public | Δ | What changed (EN docs) | Do we have it? (path / commit) | Gap / how we close |
|---|---|---|---|---|---|
| v2 (22–23.10) | 0.17023 | — | fold-ensembles pooled with NMS=max → 3–4× detections | n/a (we never pooled folds that way) | — |
| v4 (23.10) | 0.18455 | +0.014 | one fold model per expert; region-level sat fusion; cross-image dedup | have: one model per group, `post` dedup | — |
| **v5 (23.10)** | **0.33021** | **+0.146** | root cause: `laspy` absent on platform → LiDAR emitted nothing; vendored laspy; no PMF on ground-only clouds | have: own LAS reader / Rust `expds_core`, LiDAR P/R .516/.504 on LB | none |
| v6–v8 (23–24.10) | (not submitted / simulator) | — | LiDAR U-Net ConvNeXt-T, 9 relief channels, heatmap + per-class masks (0.4479 → **0.5223** with mask components for hillforts/forts); dual emission damaged/intact (0.5223 → 0.5319); 2 seeds per fold (0.5342 → 0.5489); LiDAR→sat/Ae twins α 0.1 (0.348 → 0.365) | masks: `li_mask`, `li_fort` ✔; dual hedge ✔ (tested/closed); twins `kurgan_twins_only` ✔; ConvNeXt U-Net = our cnx (kernel `convnext_main.py`) **not deployed** (`[li_cnx] enabled=false`) | G6: our ConvNeXt line never beat li_r18_hard2 in stack; root causes in §2 |
| **v9 (24.10)** | **0.41693** | +0.087 vs v5 | sat fusion D-FINE + RT-DETRv2 + heatmap (+ local D-FINE no-DN as 4th member, 0.3023 → 0.3068); Ae on | we are ≈ here (0.41238) with a *homogeneous* sat stack | G1 |
| v10 | — | — | LiDAR 3-model (s0, s1, LI_AUG); Ae off | Ae kurgans off ✔ | G2/G3/G5 |
| **v11 (24.10)** | **0.44032** | +0.023 | 4-model LiDAR blend incl. **augx** (copy-paste + openness±/SVF 8 dir 10 m, relief from DTM per crop), thr 0.15 | partial: copy-paste ✔, height/zoom ✔ but *post-hoc on u8*, svf_r/opp_r only in closed c5 | G2/G4/G5 |
| **v12 (25.10)** | **0.48874** | **+0.048** | sat 6 members (D-FINE ×3 seeds/recipes incl. site-balance+quality-aug, RT-DETR, heatmap), arch thr 0.15, **framing prior**, damaged/intact hedge 0.4; MagGr objects 0.75; **LiDAR d8 TTA full tiles** | framing = FR1h (waits «да»); hedge ✔; cnx d8 ✔ (tta=8); heterogeneous sat members ✘ | **G1 (largest)**, G9 |
| v13 (25.10) | 0.48634 | −0.002 | + LiDAR neighbourhood prior (worked: LiDAR R +0.039), retrained sat heatmap (did not transfer), MagGr 0.6 (FP) | neigh prior code ✔ off | G7 |
| **v14 (25.10)** | **0.48906** | +0.0003 vs v12 | v12 + neighbourhood prior + geo pole anchors (MagGr 0.75) | pole: `poly_mode="pole"`, `multipole_object_poly=true` ✔ | — |
| v15–v17 (25.10) | 0.48678 (v17) | −0.002 | hillfort snap, LiDAR→Ae twins (did not fire) | twins ✔; snap — not needed | — |
| **v19b (26.10)** | **0.49890** | +0.0098 | EtGr layer isolation + one geophysics pass per region (0d84268) | ✔ in eb268d6 | — |
| **v20 (27.10)** | **0.50190** | +0.0030 | Ae fortifications via FCOS (609de3b) + sat source-layer preservation (neutral) | ae_fort ✔ | G11 |
| v22 (27–28.10) | **0.50491** | +0.003 | small rules (LORO-safe); same detectors | — | — |
| v23/v24 (28–29.10) | (ledger 0.537 rehearsal, no public in docs) | — | DINOv3-SAT hillfort localiser (c58d053, 3e12e1b) | banned | G10 |
| 29.10 rejected | — | 0 | raw LAS returns (AUC 0.4127 alone), WiSE-FT, LSKNet, DINOv3 arch, partial-label pseudo-kurgans (−0.0133), masked loss (−0.0029), 8-view detector TTA on v26 (0.5306 → 0.5271) | — | do NOT repeat |

Notes on attribution: EN's *two* big model jumps are v11 (LiDAR augx blend) and v12 (satellite heterogeneous fusion + framing).
Everything after v14 is ≤ +0.01 each and mostly geophysics/Ae, which we already have.

## 2. Why our ConvNeXt LiDAR line (cnx c3/c5, li_hm, r50c5) failed where EN's worked — concrete code-level differences

Read from our trainer `repo/kaggle/2026-10-28/convnext_lidar/src/convnext_main.py` (same recipe in w2 kernels):

1. **Height aug in u8 space after clipping** (L449–452): `img[c] = 128 + (img[c]-128)*h` on lrm_s (clip ±0.75 m) / lrm_l (±2 m).
   Mounds taller than the clip are already flat-topped; ×2.2 makes the plateau bigger, ×0.5 makes a clipped mound look like a
   0.37 m mound with a flat top — a shape that never exists. EN recomputes relief from a height-scaled DTM (EXPERIMENTS
   "relief recomputed per crop from the DTM"), so clip saturation is applied *after* scaling, physically.
2. **Zoom by bilinear resize of relief** (L417, L428–433): local-relief windows (5/15 m) were computed at native scale; resizing
   changes their effective metric radius ×s. EN zooms the DTM and recomputes relief with fixed metric windows.
3. **No site balancing** (L401–403): object windows sampled ∝ class weight → dense sites dominate (EN WEAKNESSES #21 same
   diagnosis; site-balance +0.012–0.016).
4. **Additive deployment against li_r18_hard2 + dedupe 12 m** (`[li_cnx]` comment block): new detections only count where
   li_r18 has none; EN deployed the U-Net blend *as* the LiDAR expert. Our replacement test (P-W2-6 b) used a 3-seed
   *same-recipe* average, which EN also found weak (augx_s1 "no gain, large seed variance").
5. **Threshold instability across families** (MASTER_PROMPT_CLAUDE_2026-10-29: one net needs 0.329 vs 0.064 by family):
   recipe-diverse blends + neighbourhood prior are EN's practical fix (one thr 0.15 chosen in all folds).

## 3. Closure plan under FULL GO (t225u) — ordering

1. **Now (lidar-aug executor, branch `bots/a-lidar-aug`, wt `wt/lidar-aug`)**: B1 relief-recompute aug + B2 site-balance in one
   recipe ("R-aug"), f0/f1 early-stop, then 5 folds. Spec: `PROMPT_LIDAR_AUG_SEEDS_20261101.md`.
2. **In parallel, 2nd GPU**: SAT2 S0 (spor_focal d4-TTA OOF, inference only) → S1 second detector family. Spec: `EN_PLUS_WORLD_SOTA_20261101.md §5`.
3. After R-aug passes early stop: R-augx (B3 channels) and R-ms (B4 multi-scale crop), 1 seed each → recipe blend (B5) + neigh prior (B6).
4. FR1h LB, any LB upload — only with Elena «да» (bot never submits).
