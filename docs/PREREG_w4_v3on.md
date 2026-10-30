# PREREG W4 v3_on gate (bot5-w4, 30.10.2026 ~16:10 MSK), written BEFORE any v3 measurement

Question: does configs/v3_on.toml (v2_clean + [sp_dfine] variant R + [post.frame_prior]) beat the v2clean base on honest-10?

## Data / engine (same as W1 D3 gate, orch/bot-W1/work/d3_gate.py, imported unchanged)
* Metric: wt/w1/materials/compute_metrics.py via expds.metric (scope global), TableEngine on w5data/work/table.pkl.
* Folds: M.family_folds(regions, site_family, 10); honest = per-fold resweep of group thresholds on the other 9 folds, unseen=drop.
* Bootstrap: M.paired_bootstrap_counts by family, 2000 resamples, seed 0.
* Slices: all / new34 (notes/regions_las_new.txt) / nolidar (GT regions without any li object).

## Base B0 = v2clean
table A minus the 403 rows of markup etgr (regions 200-217) plus the 1578 etgr rows (threshold_group etgr_org) of
w4geo/eval/C19etonly_orget_s0.geojson (= the v2clean PIXGBM-clean + org_etgr output; maggr 3878 / terfl 319 equal A).
Sanity S1: A - etgr(table) + etgr rows of A18.geojson must reproduce A honest 0.3146153227 (extras path check).
Sanity S2: B0 vs B0: delta = 0 exactly.

## Variants
* V3P (PRIMARY, decides the gate): B0 minus all rows of groups sp_or/sp_focal/sp_arch/sp_focal2 on regions covered by the
  D-FINE SpOr OOF (bot-9 koz3-dfine-spor-full folds 1,2,4 + koz3-dfine-spor-full-rest folds 0,3; 132 regions = all 132 regions
  with sp groups in the table, so frame_prior (acts only on sp groups) is a no-op), plus sp_dfine objects:
  OOF ellipses (conf >= 0.05) -> expds/sp_dfine.cluster_circles per (region, class) over all rasters of the region
  (radius 4 m, merge 4 m, ensemble max; identical AST to the deploy code) -> post.merge_within_markup -> post.finalize,
  markup sp_or, threshold_group sp_dfine. Joint resweep of ALL groups incl. sp_dfine.
* V3F (secondary, informational): as V3P but sp_dfine thresholds fixed to [thresholds_by_markup.sp_dfine] of v3_on.toml
  (kurg_povr 0.555265, kurg_tselye 0.670285, gor 0.68847, arkh 0.373015, fort 0.869051), other groups resweep.
  Note: these thresholds were tuned by bot-9 on the same OOF (not nested) -> optimistic, hence secondary only.
* or.arkhitektury = 0.483891 in v3_on is irrelevant here (groups are resweept).

## Acceptance (P4), on V3P vs B0
delta_all >= +0.003 AND P(V3P > B0) >= 0.9 AND per-class F1 floors (>= -0.002 kurgany_povr/tselye/gorodishcha, >= -0.005 others)
AND delta new34 >= 0 AND delta nolidar >= 0. One test, no multiplicity correction. Reference (not a criterion): bot-9 R vs A +0.0198, P 0.976.

## Known deviations
OOF = fold models (final model dfine_f0 differs); deploy assumes every SpOr raster finishes (complete=True) so drop_groups hits all 132 regions.
