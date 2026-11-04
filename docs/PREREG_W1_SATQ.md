# PREREG W1-SATQ (02.11.2026, bot-W1-satq; committed BEFORE any SATQ measurement)

Spec: orch/COMPETITOR_GAPS_2026-11-02.md (d2fe1de) §2 #3/#4, §3 Q1/Q2 + §0 frame_prior finding; brief from bot 1 (02.11).
Clean-room: no EN code, nothing from bots/bot3-*; only the ideas/numbers listed in the spec.
Q2 (framing gate on large scenes) is NOT part of this family: it waits for W4's Q0 (new34 loss by scene size).

## Context, base, gate (unchanged from SATREC2/SATDET)
- v3_on b447649, production replay PH (read-only copy of b447649, configs/v3_on.toml), honest-10 nested over
  T["folds"] (orch/bot-W1/work/v3prod/t_taus.json). Base PH 0.345439 ± 0.0005; gate-side base V3P 0.3337867 ± 1e-6 (asserted).
- **One family, m = 5 variants (Q1a Q1b Q1c FP1 FP2). Bonferroni P ≥ 1 − 0.1/5 = 0.98** (paired family bootstrap p_gt0,
  2000, seed 0, D.site_family, on production-replay counts).
- **Q1 variants additionally need P ≥ 0.99** (class chosen post hoc, see below) ⇒ Q1a/Q1b/Q1c threshold = 0.99.
- PASS ⇔ Δall ≥ 0.003 ∧ P ≥ threshold ∧ Δnew34 ≥ 0 ∧ Δnolidar ≥ 0 ∧ P4 per-class F1 floors (kp/kt/gor −0.002, others −0.005);
  all measured on the production replay output.

## Q1 — arkh only: scene-centre prior + recall-oriented threshold (sp_dfine)
- **Post-hoc disclosure: the class arkhitektury was chosen AFTER seeing SATREC C1/C2, SATREC2 A1–B3 and SATDET P1/P2
  results** (arkh gained in all of them; the failures were new34 via gor and the kp/kt floors). Hence P ≥ 0.99.
  Note: new34 has 0 arkh GT (counts_B0), so Δnew34 = 0 is expected by construction unless post-steps interact; nolidar
  holds 406/407 arkh GT.
- Prior (`[post].sat_center_prior`, expds/sat_recall.py, flag default OFF): m(d) = max(0, 1 − s·clip((d − d0)/(1 − d0), 0, 1)),
  d = scene_drel (sp_or frame boxes), group sp_dfine, **s_by_class = {arkhitektury: s}, s = 0 for every other class**.
  Grid d0 ∈ {0, 0.25, 0.5}, s ∈ {0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5}.
- Recall threshold: τ(sp_dfine, arkh) ∈ {0.15, 0.20, 0.25, 0.30, 0.373} (spec grid; replaces the honest refit for that one
  group/class only; all other (group, class) τ = honest refit on the same 9 training folds at the same prior point).
- **Q1a**: prior only, (d0, s) nested, honest τ. **Q1b** (the Q1 lever): prior + τ grid, (d0, s, τ) nested jointly.
  **Q1c**: τ grid only (s = 0), τ nested.
- Nested selection per held fold k: argmax of the gate score on the 9 training folds; ties → smaller s, then larger τ,
  then larger d0. All other classes bit-identical to base (asserted gate-side).

## FP — frame_prior rebound to D-FINE (bug fix of v3_on no-op)
- Finding (§0): v3_on `[post.frame_prior]` groups = sp_or/sp_focal/sp_focal2, which `sp_dfine.drop_groups` removes on
  D-FINE regions ⇒ no effect there. Fix = same multiplier on group **sp_dfine**:
  m(d) = 1 − (1 − edge)·clip((d − d0)/(1 − d0), 0, 1) for d > d0 (post.frame_prior form), classes **fixed = v3_on list ∩
  D-FINE classes = {kurgany_povrezhdennye, kurgany_tselye, arkhitektury, fortifikatsii}** (gor excluded, as in FR1h).
  Placement: `[post].sat_center_prior` legacy spec {groups=["sp_dfine"], classes, d0, edge} → post.frame_prior, flag
  default OFF; the existing `[post.frame_prior]` (sp_or groups) is left unchanged.
- **FP1**: v3_on parameters as is (d0 0.5, edge 0.5), no selection, honest τ refit.
- **FP2**: (d0, edge) nested, d0 ∈ {0.25, 0.5, 0.75}, edge ∈ {0, 0.25, 0.5, 0.75, 1.0 (= identity)}, one pair for all 4
  classes; ties → larger edge, then larger d0. Honest τ refit. Disclosure: the form overlaps SATREC C1 (all classes, FAIL
  on kt/new34); FP keeps the v3_on class list fixed a priori.

## Production replay specifics
- The prior is applied in the replay exactly at the production location (watchdog._finish_v3: after rerank_arch /
  neighbour prior, right before post.frame_prior), via a wrapper of post.frame_prior in the replay only.
- Per-fold chosen parameters and τ go into the fold config (thresholds_by_markup.sp_dfine).

## Reporting
Per variant: Δ (all / new34 / nolidar), P raw, verdict vs its threshold, class floors, chosen parameters per fold;
commits; Q0 status → orch/bot-W1/STATUS.md.
