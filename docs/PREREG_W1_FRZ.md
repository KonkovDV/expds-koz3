# PREREG W1-FRZ — re-gate C1 / A1 / P2 with non-satellite thresholds frozen (02.11.2026, bot-W1-satq)

Committed BEFORE any FRZ measurement. Clean-room (no EN code, nothing from bots/bot3-*). Separate family from W1-SATQ
(docs/PREREG_W1_SATQ.md 0614536, which stays as committed; its freeze/Q2 amendment will be a separate commit).

## Why / disclosure
W4 Q0 (bots/w4-v3on 7c33027, PREREG 17f15d8, docs/bots/2026-11-02/bot-W4/q0_new34/) attributed the new34 losses of
C1 (SATREC), A1 (SATREC2) and P1/P2 (SATDET track 2) to nested re-tuning of NON-satellite thresholds (gor_phys, li_coarse
re-picked when sp_dfine gor scores change), not to the satellite lever. This design was made AFTER seeing those results
and the W4 breakdown, which used new34 labels; new34 is therefore not a blind check for this family. No new34 / nolidar
label enters selection below.

## Variants (lever definitions unchanged; only the threshold rule changes)
Prior on group sp_dfine before thresholds: m_c(d) = max(0, 1 − s_c·clip((d − d0)/(1 − d0), 0, 1)), d = scene_drel
(bot-10 frame_boxes.json sp_or boxes, representative point, min over covering boxes; no box ⇒ m = 1).
- **C1F** (= SATREC C1): one s for all sp_dfine classes; d0 ∈ {0, 0.25, 0.5, 0.75}, s ∈ {0.25, 0.5, 0.75} + identity
  (13 points). Ties → smaller s, then larger d0.
- **A1F** (= SATREC2 A1): s_kt for kurgany_tselye, s_o for every other class, d0 shared; d0 ∈ {0, 0.25, 0.5},
  s_o, s_kt ∈ {0, 0.25, …, 1.5} (147 points), per-class separability. Ties → smaller s_kt, smaller s_o, larger d0.
- **P2F** (= SATDET P2): s_gor, s_arkh independent, s = 0 for every other class, d0 shared; same grid as A1F.
  Ties → smaller s_gor, smaller s_arkh, larger d0.

## Freeze rule (the only change)
For every outer fold k and every point: thresholds of ALL non-sp_dfine groups (gor_phys, li_coarse, li, ae, …, every
group except sp_dfine) = the base honest τ of fold k (V3P fold taus, t_taus.json), unchanged; missing stays missing.
Only sp_dfine per-class τ is re-fit on the 9 training folds: exact 1-D argmax of the training-fold class F1 over the
sp_dfine cut points (M._group_cuts, first maximum) with the other groups fixed at base τ. Identity point = base exactly
(base τ for every group). Point selection per held fold: max in-sample gate score on the 9 training folds (as originally).
Diagnostic only: whether the frozen 1-D re-fit at identity reproduces the base sp_dfine τ.

## Base, gate
- v3_on b447649 production replay PH (read-only copy, configs/v3_on.toml), honest-10 nested over T["folds"];
  base PH 0.345439 ± 0.0005, gate-side base V3P 0.3337867 ± 1e-6 (asserted). The prior is applied at the production
  location of `[post].sat_center_prior` (watchdog finish, right before post.frame_prior) via a replay-only wrapper.
- **Family m = 3 (C1F, A1F, P2F); P threshold 0.98333 (per bot 1; = 1 − 0.05/3)**, paired family bootstrap p_gt0
  (D.site_family, 2000, seed 0) on production-replay counts.
- PASS ⇔ Δall ≥ 0.003 ∧ P ≥ 0.98333 ∧ Δnew34 ≥ 0 ∧ Δnolidar ≥ 0 ∧ P4 per-class F1 floors (kp/kt/gor −0.002, others −0.005),
  all on the production replay. Anything else → FAIL. No change after measurement; no LB submit; flags default OFF.
- Production placement if PASS: `[post].sat_center_prior` {groups=["sp_dfine"], d0, s, s_by_class} + thresholds_by_markup
  sp_dfine τ; non-satellite thresholds = current v3_on values.
