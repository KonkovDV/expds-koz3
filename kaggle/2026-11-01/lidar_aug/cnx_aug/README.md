# R-augx fold0 (W2, PROMPT_LIDAR_AUG_SEEDS_20261101 B1+B2+B3) — kernel elenakonkova/koz3-lidar-aug-f0
Base recipe = c3-sc s2 reference kernel (w2/kernels/source_c3sc/koz3-w2-cnx-c3sc-s2f01), ConvNeXt-T scratch, centerhead, 12k it.
New, env-flagged (defaults off = reference code path, same rng consumption; 2026-10-28/convnext_lidar untouched):
- `CNX_AUG_RELIEF=bank` (B1): per train region, relief recomputed from transformed DTM (z05 work grid):
  height on residual z' = lowpass(z,50m) + h·(z − lowpass), zoom = bicubic XY resample of DTM, then XL.relief_channels /
  RX.extra_channels (stats of bank (1,1)) — same metric windows/clips as inference. BANK7 = {0.75,1,1.3}×{0.6,1.7}+(1,1)
  (`CNX_BANK=9` → 3×3). GT points/radii and masks scaled by zoom. Sampling: bank uniform + zoom jitter ×U(0.92,1.08) +
  lrm-only height jitter ×U(0.9,1.1) (p 0.5). Old u8 height ×0.5–2.2 / zoom 0.7–1.45 paths are OFF in bank mode.
  Parity: bank (1,1) via bank path vs inference relief asserted on 2 smallest train regions → SystemExit on mismatch (no u8 fallback).
- `CNX_SITE_BALANCE=0.5` (B2): site p ∝ n_obj^α, then object by class weight; background site p ∝ valid_px^α; per-site
  sample counts logged (20k simulated draws) into report.
- c7 jobs (B3): lrm_s, lrm_l, slope, svf_r, opp_r, opn_r (neg. openness), msrm (|TPI| 5/10/20/40 m / clips 0.25/0.5/1/2 m,
  max-scale signed). Openness via expds.accel (Rust) for c7.
Local smoke (box, 2 LAS Гнёздово_Лесная_2/3): parity base3 + ext exact (mismatch 0.0); 7 banks build in 0.2–0.7 s/region-bank.
Early stop (PROMPT §C): fold0 li-only kurgan vs c3-sc s2 f0 0.2198 (stop if < 0.2098); f1 pending.
Regenerate: `python gen_kernel.py` (needs /workspace/koz3/w2/kernels/source_c3sc). S3 creds are NOT in git (push-time inline, scrubbed).
