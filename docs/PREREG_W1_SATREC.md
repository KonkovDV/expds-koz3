# PREREG W1-SATREC — satellite recall audit + recall-first / fusion / scene-centre prior (bot-W1, 02.11 MSK)

Written and committed BEFORE any measurement of the quantities below. Clean-room: written from the task spec
(bot 1 brief 02.11, orch/COMPETITORS_2026-11-02.md ideas only); no EN code, nothing from bots/bot3-*.

## Base (B) — v3_on production replay, PH mode
Same replay as tools/v3prod (bots/w1-cfdeploy 9ea621d, docs/PREREG_W1_V3PROD.md): table OOF stack (table.pkl +
geom_wkb.pkl) minus etgr, minus sp_or/sp_focal/sp_arch/sp_focal2 in the 132 D-FINE regions (dreg = all sp
regions), + c19 (etgr) + dfx (sp_dfine: D-FINE OOF ellipses conf>=0.2 → cluster_circles 4 m → merge_within_markup →
finalize). Per outer fold f (honest-10, family_folds(regs, site_family, 10), unseen=drop): early floors = thresholds =
τ_f (honest sweep_fixed on the 9 other folds), σ 12 m rules, then watchdog.finish of archive b447649 with
configs/v3_on.toml (code extracted read-only with git archive; archive itself untouched). New in replay vs v3prod:
scene_drel annotated for every feature as production does (pipeline._with_scene_drel: post.annotate_scene_drel with
bot-10 frame_boxes.json boxes of the feature's markup_type) — needed by C1/C2; FR1h only touches groups that are
dropped in dreg, so B must not change.
Scoring: official greedy matching M.build_curves(scope global) of the final outputs → per-region counts → class-weighted
F1; bootstrap M.paired_bootstrap_counts(V, B, site_family, 2000, seed 0), slices all / new34 / nolidar.
Sanity S (stop if failed): B_PH with this tooling = 0.345439 ± 0.0005 (v3prod result); gate honest of base extras
reproduces V3P 0.3337867 (± 1e-6).

## Part 1 — recall-loss audit (diagnostic, no gate)
Satellite GT = GT with markup_type sp_or (1606 objects). For each outer fold and class, count sat GT objects that have
≥1 valid candidate (official _match_quality: coverage ≥ 0.5 + interior point, same class) after each stage:
 L0  union of all raw sat detectors: r18 rows (sp_or/sp_focal/sp_arch, conf ≥ 0.05) + sp_focal2 (≥ 0.05) + D-FINE raw ellipses (≥ 0.2)
 L0r r18 only; L0f sp_focal2 only; L0d D-FINE raw only
 L1  after drop_groups (v3: only D-FINE remains) and D-FINE cluster/merge/finalize (dfx)
 L2  after early score floors / thresholds τ_f (per fold, per class)
 L3  after rerank_arch / σ 12 m / apply_thresholds stage inside finish (sat-origin outputs)
 L4  after the full post chain (merge_within_markup, twins, micro_circles, filter_finds, finalize, cap) — sat-origin outputs
 L5  matched (TP) in the official greedy global matching by ANY final output; L5s matched by a sat-origin output.
Also counterfactual L1' = r18+f2 kept (no drop_groups) at their honest τ (= v2c behaviour) for comparison.
Reported: per fold × stage totals and per-class totals; sat precision/recall of B.

## Part 2 — variants (all over B; m = 5 → Bonferroni)
 R1  recall-first single τ for sp_dfine: one shared threshold for all 5 sp_dfine classes, from grid
     {0.20,0.25,...,0.70}; chosen per outer fold by max in-sample gate score on the 9 training folds (other groups'
     τ re-swept jointly with sp_dfine fixed; ties → lowest τ). Floor 0.20 = D-FINE OOF floor.
 R2  equal-weight fusion of the three sat detectors (r18 = sp_or/sp_focal/sp_arch heads, sp_focal2, D-FINE dfx) into
     one group "sp_fuse" (markup sp_or), replacing sp_dfine; per region and class, candidates in descending conf;
     each unassigned one becomes an anchor and absorbs unassigned candidates with centroid distance ≤ 8 m (ground
     metres) or centroid inside the anchor geometry or anchor centroid inside theirs. Fused conf = mean over the
     detectors that can emit the class of each detector's max conf in the cluster (absent = 0): kurgans /3 (r18,
     f2, D-FINE), gorodishcha & arkhitektury /2 (r18, D-FINE), fortifikatsii /1 (D-FINE). Geometry = anchor's.
     Per-class honest τ for sp_fuse (as any group). σ 12 m rule not applied to f2 (consumed by fusion).
 R3  R2 + single shared τ for sp_fuse from grid {0.05,0.075,0.10,0.125,0.15,0.20,0.25,0.30,0.35,0.40,0.50},
     nested like R1.
 C1  scene-centre prior on sp_dfine (before thresholds, = post.frame_prior form applied to group sp_dfine):
     m(d) = 1 − s·clip((d − d0)/(1 − d0), 0, 1), d = scene_drel (min over covering sp_or scenes; None → 1);
     width d0 ∈ {0, 0.25, 0.5, 0.75}, strength s ∈ {0.25, 0.5, 0.75} plus identity (s = 0): 13 points, chosen per
     outer fold by max in-sample training-fold gate score with τ re-swept per point (ties → smaller s, then larger d0).
 C2  C1 prior (same grid, nested) applied to sp_fuse of R2 (per-class τ).
Selection of every hyperparameter uses only training folds of the gate (stack curves); the outer held fold is scored
only in the production replay. Chosen values per fold are reported.

## Gate (per variant V vs B, PH replay, held-out folds)
PASS iff all: Δ_all ≥ 0.003; P(Δ>0) ≥ 0.98 (Bonferroni 1 − 0.1/5; raw P also reported);
P4 class floors F1_V ≥ F1_B − 0.002 (kurgany_povrezhdennye, kurgany_tselye, gorodishcha), − 0.005 (others);
Δ_new34 ≥ 0 and Δ_nolidar ≥ 0. Anything else → FAIL. No variant is changed after measurement; no LB submit.
Production code (if any PASS): flags on bots/w1-satrec, default OFF.
