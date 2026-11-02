# PREREG W1-COMBO — C1F + FP1 (02.11.2026, bot-W1-satq; committed BEFORE measurement)

Request: bot 1 (02.11, after W1-SATQ e574579). Both parts already passed alone: C1F (FRZ 5cc5ed7, +0.01592, P 1.0) and
FP1 (SATQ e574579, +0.008581, P 0.999). This gate tests whether they stack. **Seen before this PREREG:** both single-lever
results and every number in RESULTS_W1_FRZ / RESULTS_W1_SATQ. No combo number has been computed.

## Variant COMBO (one variant)
- Production order on sp_dfine, before thresholds: **C1F centre prior first** ([post].sat_center_prior {groups=[sp_dfine],
  d0, s=0, s_by_class = s for every D-FINE class}), **then FP1** (frame_prior on sp_dfine, classes kp/kt/arkh/fort, d0 0.5,
  edge 0.5, i.e. m = 1 − 0.5·clip((d − 0.5)/0.5)), then the v3_on [post.frame_prior] as is (no-op on sp_dfine).
  Both are multiplicative in confidence; the order is fixed anyway.
- C1F (d0, s) nested exactly as in FRZ: identity (s = 0) + d0 ∈ {0, 0.25, 0.5, 0.75} × s ∈ {0.25, 0.5, 0.75}, one s for all
  D-FINE classes; ties → smaller s, then larger d0. The identity candidate = FP1 alone. FP1 fixed at (0.5, 0.5), not selected.
- Freeze rule (FRZ 1532475): non-sp_dfine τ = base fold τ; sp_dfine per-class τ = exact 1-D argmax of training F1
  (first max, 6 decimals), refit at every candidate point.

## Gate
- v3_on b447649 production replay (configs/v3_on.toml), honest-10 nested over t_taus folds; base PH 0.345439 (± 5e-4),
  gate-side base 0.3337867 asserted. Paired site-family bootstrap, 2000 resamples, seed 0.
- **m = 1** for this decision. Honest context: this is the 3rd W1 family on the same data (FRZ m = 3, SATQ m = 6) and the
  variant combines two winners chosen from those families, so it is not an independent test. To stay consistent with FRZ the
  threshold is kept at **P ≥ 0.98333** (= 1 − 0.05/3), stricter than the m = 1 value.
- PASS ⇔ Δall ≥ 0.003 ∧ P ≥ 0.98333 ∧ Δnew34 ≥ 0 ∧ Δnolidar ≥ 0 ∧ P4 floors (kp/kt/gor −0.002, others −0.005).
- For information only (no decision): Δ and P of COMBO vs C1F alone (frz/out_C1F.geojson) and vs FP1 alone
  (frzq/out_FP1.geojson), same bootstrap.

## If PASS — production config
Flag [post.w1_c1f_fp1] (default enabled = false): C1F d0/s and FP1 d0/edge/classes + own sp_dfine thresholds, selected ONCE
on all regions with the full-data C1F rule (tools/satq/full_prep.py); non-sp_dfine thresholds untouched. Expected PH
(in-sample production replay) written to the handoff docs/W1_FP1_HANDOFF.md.

Outputs: /workspace/koz3/w1/satq/combo/ (new dir; nothing earlier overwritten).
