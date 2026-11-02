# PREREG W1-SATREC2 (02.11.2026, bot-W1-satrec; committed BEFORE any SATREC2 measurement)

Follow-up to W1-SATREC (docs/RESULTS_W1_SATREC.md, 06ef4ec: R1 and C1 cleared Δ/P but failed new34; C1 also failed the kt
floor). Disclosure: this design was made after seeing those results; new34 LABELS are not used anywhere in selection
(new34 membership is used only as an exclusion list for H, see below).

## Context, base, gate (unchanged)
- v3_on b447649, production replay PH (tools/v3prod on bots/w1-cfdeploy 9ea621d), honest-10 nested. Base B = v3_on PH
  (expected 0.345439 ± 0.0005). Gate-side base = V3P 0.3337867 ± 1e-6.
- **One family, m = 6 variants (A1 A2 A3 B1 B2 B3). Bonferroni: P ≥ 1 − 0.1/6 = 0.98333** (paired family bootstrap
  p_gt0, 2000, seed 0).
- PASS ⇔ Δall ≥ 0.003 ∧ P ≥ 0.98333 ∧ Δnew34 ≥ 0 ∧ Δnolidar ≥ 0 ∧ P4 per-class F1 floors (kp/kt/gor −0.002, others −0.005).

## Lever
Centre prior on group sp_dfine before thresholds: m_c(d) = max(0, 1 − s_c · clip((d − d0)/(1 − d0), 0, 1)),
d = scene_drel (sp_or scene boxes, representative point, min over covering boxes); no box ⇒ m = 1.
Per-class strength: s_c = s_kt for kurgany_tselye, s_c = s_o for all other classes; d0 shared.
- Grid: d0 ∈ {0, 0.25, 0.5}; s_o, s_kt ∈ {0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5} (147 points, identity included).
- B adds a single sp_dfine τ ∈ {0.20, 0.25, …, 0.70} (11) applied after the prior (fixed group, as SATREC R1).
- Selection per held fold k: maximise gate (stack, honest machinery) score on the 9 training folds. Ties: smaller s_kt,
  then smaller s_o, then larger d0, then smaller τ. Per-class decomposition (score = Σ w_c F1_c, classes independent)
  is used to evaluate the s_o × s_kt combinations exactly.

## Variants
- **A1 C1+**: prior, per-class strength, extended grid, plain nested selection.
- **A2 C1+ / modality rule** (a priori rule): prior applied only in regions with NO lidar candidate entering finish
  (markup li, after output floors; = 158/210 regions, 122/132 dreg; list docs/results_w1_satrec2/nl_regions.json "nl_post").
  sp_dfine keeps per-class honest τ.
- **A3 C1+ / constrained selection**: as A1, but admissible only if on the training folds (i) every class F1 Δ vs base ≥
  its P4 floor and (ii) Δscore on H ∩ training folds ≥ 0. H = new34-like holdout defined a priori without new34 labels:
  dreg regions that also have a non-satellite modality (li/ae/or candidates), minus the new34 regions (8 regions,
  docs/results_w1_satrec2/H_regions.json). Identity is always admissible.
- **B1 R1+C1**: single τ + prior (147 × 11), plain nested.
- **B2 R1+C1 / modality rule**: in NL regions sp_dfine → group sp_dfine_nl with single τ + prior; elsewhere sp_dfine
  keeps per-class honest τ, no prior.
- **B3 R1+C1 / constrained selection**: as B1 with constraints (i)+(ii); base (per-class τ, no prior) is the fallback
  candidate and is always admissible.

## Production placement (flags default OFF)
`[post].sat_center_prior` in watchdog.finish (before frame_prior) gains: s_by_class, s > 1, `only_without_lidar`
(region has no markup-li feature in the finish input). B2 group split: `[post].sat_center_prior.nl_group`.

## Reporting
Per variant Δ (all/new34/nolidar), P raw, Bonferroni verdict, class floors, chosen hyperparameters per fold.
