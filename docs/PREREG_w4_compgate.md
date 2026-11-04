# PREREG P-W4-compgate (+ v3_on + M_rp) — W4 bot5, 30.10.2026 16:10 MSK, written before any measurement

Engine: same honest-10 as the v3_on gate (orch/bot-W1/work/d3_gate.py imported unchanged, table.pkl, M.family_folds(10),
joint resweep, paired family bootstrap 2000 seed 0, slices new34 / nolidar, floors FLOOR_TIGHT −0.002 / other −0.005).

## R. Rebuild of OOF stack A on the box (PC offline)
Fresh split + fresh cache (w4geo/compgate/split_arch_R, cache_R) from w5data/stack_arch.geojson + gt_train_v2.geojson,
curves rebuilt by the official compute_metrics matching (Stack._curves_A, no cache reuse).
PASS iff honest(A_R) == 0.3146153227 (|d| ≤ 1e-9) and every cluster curve blob equals the cached A (S.blob).

## C. Composition-weighted gate (#12, REVIEW_SOTA)
- Thresholds: the unweighted honest joint resweep per fold (fold_taus); counts are then split by modality with the
  same frozen fold thresholds, so Σ_mod counts == unweighted counts (sanity C0, must be exact).
- Modality of a prediction = its threshold group: li, li_dev, li_mosaic, li_coarse, li_mask, li_fort, li_ae → li;
  sp_or, sp_focal, sp_focal2, sp_arch, sp_dfine → sp_or; ae, ae_fort → ae; or → or; maggr, etgr, terfl, gor_phys → geo.
  Modality of a GT object = markup_type (maggr/etgr/terfl → geo). TP is attributed to the prediction's modality.
- Weights w_m = target_share_m / OOF_GT_share_m, applied to (tp, fp, n_gt) of modality m; score = official class-weighted F1.
  PUB = public test GT counts (REVIEW_SOTA §2.1): geo 1441, li 964, sp_or 377, ae 75, or 50.
  CLO ("closed" scenario) = equal share 1/5 for each of the 5 modalities (or has only 184 OOF GT → noisy; disclosed).
- Sanity C1: w ≡ 1 reproduces the unweighted score exactly.
- Reported for every comparison: Δ and P (family bootstrap on weighted counts) for UNW, PUB, CLO.
- Tool rule (#12): a variant is "composition-OK" iff Δ_PUB ≥ 0 and Δ_CLO ≥ 0.

## Variants
1. V3P vs B0 (v2clean) — the already-accepted v3_on, re-read through PUB/CLO (no new acceptance; diagnostic).
2. MRP on A vs A — M_rp as defined by W3 (orch/bot-W3/prereg/C5B_JOINT_PREREG.md, code/make_joint.py): every row of
   gorodishcha / arkhitektury / object_poly / fortifikatsii → disk r = 0.05 m (Mercator-scaled) at representative_point;
   thresholds re-swept. Sanity M0: the same rows re-injected as extras WITHOUT the disk transform must reproduce A
   (|d| ≤ 1e-9). Reference: W3 MRP +0.004670, MASTER_PLAN +0.0044 (P 0.989). Expected within ±0.001 of +0.0047.
3. V3P+MRP vs V3P and vs B0: MRP applied to all RPC-class rows of the V3P stack (stack rows kept by V3P, C19 etgr rows,
   D-FINE circles of gorodishcha/arkhitektury/fortifikatsii).
   Acceptance (V3P+MRP vs V3P): Δ ≥ +0.003, P ≥ 0.9, new34 and nolidar Δ ≥ 0, floors OK, and composition-OK (Δ_PUB ≥ 0, Δ_CLO ≥ 0).
   If Δ ∈ [0, 0.003): M_rp is reported as overlapping with v3_on (not stacked).
No other variants, no threshold tuning outside the joint resweep. Seeds/folds: single fold assignment (family_folds 10).
