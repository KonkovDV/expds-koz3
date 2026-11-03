# PREREG W4 — C1F-gor/arkh (GA), FP1 re-gate, and GA+FP1 (02.11.2026 MSK, bot5-W4; committed BEFORE any measurement)

Base: v3_on b447649 (PH 0.345439; gate-side base 0.3337867, both asserted). Code: b447649 + W4 C1F port 98cdf56
(incl. the scene_drel fix for sp_dfine) + a new production flag (below). Pipeline = the W4 C1F re-gate
(docs/bots/2026-11-02/bot-W4/c1f_regate, PREREG e6b5a31): gate-side nested selection on honest-10 (t_taus folds),
then the decision on a production replay through the ported code (W1 s2-type replay, s3 gate scoring, paired
site-family bootstrap 2000, seed 0).

## Disclosure
- **GA's classes (gorodishcha, arkhitektury) are chosen post hoc**, after the W4 C1F re-gate FAIL (5d0dd4c), which
  showed the loss in kurgany_tselye / new34 and the gains in gorodishcha (+0.055) and arkhitektury (+0.037).
- Seen before this PREREG: W1-FRZ (5cc5ed7), W1-SATQ results (e574579: FP1 PASS +0.00858 P 0.999; FP2 / Q2 FAIL),
  W4 C1F re-gate (5d0dd4c), incl. the per-class deltas of C1/P2/C1F_frozen.

## Freeze rule (all variants)
Per fold: every non-sp_dfine (group, class) τ = base fold τ (taus_V3P). sp_dfine τ of the classes the variant does NOT
touch = base fold τ. sp_dfine τ of a touched class: 1-D argmax of the training-fold F1 over that class's sp_dfine cut
points, all other τ fixed (highest τ on ties; "none kept" → 1.01).

## Variants
Prior (scene-centre, on group sp_dfine, before thresholds): m = max(0, 1 − s·clip((scene_drel − d0)/(1 − d0), 0, 1))
(= post.frame_prior with edge = 1 − s). Gate side: drel from bot-10 frame_boxes.json (as in the C1F re-gate).
- **GA (C1F-gor/arkh)**: prior only on gorodishcha and arkhitektury, one (d0, s) for both, nested per fold over
  d0 ∈ {0, 0.25, 0.5, 0.75} × s ∈ {0.25, 0.5, 0.75} plus identity; ties → smaller s, then larger d0. Touched τ: gor, arkh.
  kp / kt / fort and everything outside sp_dfine frozen at base.
- **FP1 (W1 e574579, as is, no re-selection)**: frame_prior form on sp_dfine for classes kurgany_povrezhdennye,
  kurgany_tselye, arkhitektury, fortifikatsii, d0 = 0.5, edge = 0.5 (s = 0.5) fixed. Touched τ: those 4 classes.
- **GAFP (combination; computed only if GA and FP1 both PASS)**: both priors (multipliers multiply on arkhitektury);
  GA (d0, s) per fold = GA's nested choice (no new selection), FP1 fixed; touched τ: all 5 sp_dfine classes.

## Production flag (default OFF)
`[post.w4_sat]` {enabled, groups = ["sp_dfine"], center = {d0, s_by_class}, frame = {classes, d0, edge},
thresholds_sp_dfine}. Resolved inside watchdog.finish / safe_finish (like the W1 flags) into `[post].sat_center_prior`
(center) and a new `[post].sat_frame_prior` (frame; post.frame_prior on the sp_dfine group, right after sat_center_prior),
plus the listed [thresholds_by_markup.sp_dfine] classes. Mutually exclusive with w1_c1f / w1_p2f. With it on, sp_dfine
features get scene_drel (the 98cdf56 fix condition is extended). Replay = this code with the flag on and the fold's values.
Effect check (FP1 / GA): the replay must change sp_dfine outputs vs base (else the variant is reported as a no-op FAIL),
and a unit test checks that `_sp_dfine_phase` annotates scene_drel with w4_sat on.

## Criteria (production replay, honest-10)
PASS ⇔ Δall ≥ +0.003 ∧ P ≥ bar ∧ Δnew34 ≥ 0 ∧ Δnolidar ≥ 0 ∧ per-class F1 floors (kp / kt / gor −0.002, others −0.005).
Bars (Bonferroni 1 − 0.1/m over the sat-prior looks: W1-FRZ C1F/A1F/P2F, W4 C1/P2, GA, FP1 → m = 7):
- **GA: P ≥ 0.985714** (m = 7).
- **FP1: P ≥ 0.985714** (same family; W1's own pre-registered bar was 0.98333 — reported too).
- **GAFP: P ≥ 0.9875** (m = 8).

## v4
If GAFP PASS → v4 = GAFP; else the single passer with the larger replay Δall; none → nothing built.
v4 values = the same rules on all 110 training regions as one training set (GA (d0, s) and touched τ), written into
`[post.w4_sat]` enabled = true in the v4 build config (v3_on.toml + flag); everything else = v3_on b447649.
