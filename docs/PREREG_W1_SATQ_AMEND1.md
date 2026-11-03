# PREREG W1-SATQ — Amendment 1: freeze rule + Q2 (02.11.2026, bot-W1-satq; committed BEFORE any frozen SATQ measurement)

Amends docs/PREREG_W1_SATQ.md (0614536). Everything not changed here stays as in 0614536.

## A. Disclosure (what was already seen)
- The original (unfrozen) **gate-side** nested prep of SATQ was run before this amendment (tools/satq/q1_prep.py,
  /workspace/koz3/w1/satq/out/prep.pkl). Gate-side nested Δ vs V3P 0.3337867: **Q1a +0.003118** (mostly d0 0.25, s 1.5),
  **Q1b +0.003203** (τ_arkh mostly 0.30), **Q1c +0.000072**, **FP1 +0.007198**, **FP2 +0.003893**. No SATQ production replay
  and no SATQ bootstrap was run. Those numbers are superseded and NOT reported as results.
- Seen before this amendment: W4 Q0 (origin/bots/w4-v3on 7c33027, new34 used: R1 losses in 035_Шумгора#s1 8.4 Mpix and
  040_Примаки#s0 5.7 Mpix; C1/A1/P new34 losses come from non-sat τ re-tuning), W1-FRZ results (5cc5ed7: C1F/P2F PASS,
  A1F FAIL), and the full-data C1F/P2F selection (af6b827). **The Q2 size cutoff A = 4 Mpix is taken from W4's Q0 (which used
  new34 labels), i.e. chosen post hoc**; it is fixed (not nested).

## B. Freeze rule (all variants; same as PREREG FRZ 1532475)
Per fold, every non-sp_dfine (group, class) τ = base fold τ (honest base). sp_dfine τ: per class, exact 1-D argmax of the
training-fold F1 over the group cut points (first maximum, 6 decimals) with all other τ fixed; at identity this reproduces
the base τ. Q1b/Q1c: τ(sp_dfine, arkh) = grid value; other sp_dfine classes by the 1-D refit (= base, their curves are
unchanged). "honest τ refit" in 0614536 is replaced by this rule for Q1a/Q1b/Q1c/FP1/FP2/Q2.

## C. Q2 — framing gate on large scenes (added to the family)
- Group sp_dfine, classes kurgany_povrezhdennye, kurgany_tselye, gorodishcha, arkhitektury (fortifikatsii untouched).
- Scene of a detection = the covering sp_or scene box that gives its scene_drel (minimum over covering boxes; first on ties);
  Mpix = (W·cos φ / 0.5) · (H·cos φ / 0.5) / 1e6 of that box, φ = latitude of the box centre (EPSG:3857; W4 Q0 definition).
  Gate side: bot-10 frame_boxes.json; production: map_footprint of each sp_or raster (same boxes, as for scene_drel).
- Rule: on scenes with Mpix ≥ **A = 4** (fixed), a detection with scene_drel > x and confidence < t2 is **removed**
  before thresholds (production location = the [post].sat_center_prior step). Detections without a covering scene or on
  scenes < 4 Mpix are untouched.
- Nested grid x ∈ {0.4, 0.5, 0.6}, t2 ∈ {0.35, 0.5, 0.7, 0.85}, plus identity (no gate). Ties → identity, then smaller t2,
  then larger x. Freeze rule (B) for τ. Production: `[post].sat_frame_gate` in expds/sat_recall.py (flag default OFF),
  scene_mpix annotated next to scene_drel.

## D. Family and thresholds
- **Family m = 6 (Q1a Q1b Q1c FP1 FP2 Q2). Bonferroni P ≥ 1 − 0.1/6 = 0.98333** (same 0.1/m convention as 0614536).
- Q1a/Q1b/Q1c additionally need **P ≥ 0.99** (unchanged).
- PASS ⇔ Δall ≥ 0.003 ∧ P ≥ threshold ∧ Δnew34 ≥ 0 ∧ Δnolidar ≥ 0 ∧ P4 floors (kp/kt/gor −0.002, others −0.005),
  on the production replay of v3_on b447649 (base PH 0.345439 ± 5e-4, gate-side base asserted 0.3337867).
- Outputs in a new directory /workspace/koz3/w1/satq/frzq/ (the earlier out/ is kept untouched).
