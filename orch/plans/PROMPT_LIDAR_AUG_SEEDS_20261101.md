# PROMPT lidar-aug + seeds — LiDAR kurgan retrain with EN ideas (not code) + beyond-EN (t225u/t227u/t228u) 01.11 ~12:05 MSK
Owner A → W2 (bot-W2-lidar, B). Branch `bots/a-lidar-aug`, worktree `wt/lidar-aug`. Written by A-executor research subagent
because the file did not exist at 12:05; **LOCKED by A 01.11 11:48 MSK — this is the authoritative PROMPT.**
Hold **0.41238**. No LB submit. No push to main, no default.toml edits. No YOLO / DINOv3 / SAM3 / SegFormer mit_*. **No EN code,
weights or text copied** — EN facts below are numbers from their docs only. Background: `EN_COMMIT_GAP_VS_US_20261101.md`,
`EN_PLUS_WORLD_SOTA_20261101.md`.

## A. Why our previous cnx line failed (fix these, not the seeds)
Trainer `kaggle/2026-10-28/convnext_lidar/src/convnext_main.py` already has height ×0.5–2.2 / zoom ×0.7–1.45 / copy-paste —
**COMPETITORS_GT05 "height/zoom aug absent" is wrong**; the real defects:
1. height aug multiplies the *already-clipped u8* relief (L449–452) → saturated tall mounds, non-physical shapes;
2. zoom bilinear-resizes relief computed with fixed-pixel windows (L417–433) → wrong metric scale of LRM;
3. object sampling ∝ class weight only (L401–404) → dense sites dominate, blind-site recall poor;
4. deployment additive (dedupe 12 m) vs li_r18_hard2; replacement test used same-recipe seeds (EN: seeds ≠ diversity).
EN evidence (their nested LiDAR metric): base 0.5327 → physically-correct aug + site balance 0.5490 → + paste/openness/SVF 0.5688 →
recipe blend 0.5765 → + neighbourhood prior 0.5831; LiDAR d8 0.5651 → 0.5695.

## B. Changes (all behind flags, default = old behaviour, so old kernels reproduce bit-exact)
- **B1 `--aug-relief bank`** (core): in data prep, for each training tile compute relief channels from a transformed DTM for a
  3×3 grid zoom ∈ {0.75, 1.0, 1.3} × height ∈ {0.6, 1.0, 1.7}: height acts on the residual `r = z − lowpass(z, 50 m)` as
  `z' = lowpass + h·r`; zoom resamples the DTM in XY (bicubic), then relief is computed with the **same metric windows** and the
  **same clip ranges** as inference (`XL.relief_channels`, `RX.extra_channels`), GT points/radii scaled by zoom. Training samples
  a bank uniformly + continuous jitter (height ×U(0.9,1.1), zoom ×U(0.92,1.08)) on the bank. Bank (1,1) == inference relief
  (assert parity on 2 tiles). Disk: 9× c3 u8 — check DS size; if too big use {0.75,1,1.3}×{0.6,1.7}+(1,1) = 7 banks.
  Fallback `--aug-relief online` (recompute per crop) only if bank won't fit.
- **B2 `--site-balance 0.5`**: sample site first with p ∝ n_obj(site)^α (α=0.5; α=0 uniform = EN-like ablation f0 only), then an
  object within site by class weight; background windows also site-balanced. Log per-site sample counts.
- **B3 `--chans c7`** (recipe R-augx): lrm_s, lrm_l, slope, svf_r, **pos-openness, neg-openness, msrm** (multi-scale relief
  5/10/20/40 m stacked into one channel as max-normalised |TPI| — Berganzo 2021 MSRM, Guyot 2021 e2MSTP). Implement in numpy
  ourselves (RVT definitions are public: Kokalj & Somrak 2019).
- **B4 `--multiscale-crop`** (recipe R-ms): crops 384/512/640 resized to 512 for training only; inference unchanged 512 d8.
- **B5 hard negatives (optional, after first PASS)**: bank of top-N inner-fold FP peaks (li_r18_hard2 precedent), 20 % of windows.
- Keep: copy-paste full height (EN attenuated paste was rejected), EMA 0.999, AMP, 12k it, focal centre loss, d8 TTA at OOF.
- Do NOT add: zoom-TTA, unclipped asinh relief, ConvNeXt-S, soft-F1 loss, raw LAS returns, pseudo-labels/masked loss, WiSE,
  per-site recalibration, test-time self-training (all measured negative or zero by EN).

## C. Run plan (one recipe at a time; seeds only after a recipe passes)
1. **R-aug = B1+B2, c3, init as before, seed 0**, folds f0+f1 first → **early stop**: li-only kurgan mean of f0/f1 vs c3-sc s2
   reference (0.2198 / 0.4786, mean 0.3492) — STOP if mean ≤ 0.3492 or either fold < reference −0.01.
2. If pass → f2..f4 seed 0 → OOF export (nn_oof format, same folds/family split) → gates (D).
3. In parallel on a 2nd GPU after step-1 pass: **R-augx (B1+B2+B3)**, then **R-ms (B1+B2+B4)**, seed 0 each.
4. **Blend = mean of logits of R-aug, R-augx, R-ms** before peak decode (not of detections). Prefer this over 3 seeds of one
   recipe (EN: 2nd augx seed no gain; recipe blend +0.008). Seeds 1/2 of the best single recipe only if a GPU is idle.
5. Neighbourhood prior (`expds/post.py:901 kurgan_neighbour_prior`, R 200 m, K 3, strength 0.2, on li/li_coarse/li_mosaic +
   new group) — re-gate on the NEW heatmap only; separate arm.

## D. Gates (pre-register in `orch/bot-W2/PREREG_W2_cnx.md` before measuring)
Arms: (a) additive dedupe 12 m vs li_r18_hard2; **(b-kur) replacement of li kurgan groups (primary)**; (c) b-kur + neigh prior.
honest-10 base A = 0.314615 (paired bootstrap 2000 / 103 families): Δ ≥ +0.003 **and** P ≥ 0.9; class floors −0.002
povr/tselye/gor, −0.005 others; Δnew34 ≥ −0.001; Δno-lidar ≥ −0.001. Added: leave-largest-site-out k* ≥ 5; chosen threshold
fold dispersion ≤ ×2 (old cnx failed at 0.329 vs 0.064); report blind-site recall (Novenkoe/Erentuna/Gnezdovo-type sites).
Runtime: blend + d8 must fit the 600 s/region rehearsal budget; if not, distil blend → 1 student (Hinton 2015) before packaging.
Packaging/LB only with Elena «да».

## E. Logistics
A leases GPU + kernel slug; data = new Kaggle DS (no S3 secrets). W2 reports to `orch/bot-W2/STATUS.md`; COORD one-liner per
milestone. Kill = write CLOSED + numbers, no reruns of the same config.
