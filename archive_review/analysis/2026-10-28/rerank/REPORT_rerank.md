# Reranker (LightGBM over stack candidates) — verdict 28.10 ~20:45 MSK
Base = honest-10 A 0.314615 (canon family_folds, same as EXPERIMENTS.md). Nested LOFamily: outer model on 9 folds, thresholds swept on inner OOF of train folds.
Features X2 (35): conf/logit (monotone), cls/grp/mk cat, geometry, twin, region rank/pct, neighbour density & cross-source consensus, NMS context. Label = first-hit under base ordering, pool = top15%/group.
| variant | canon Δ (boot mean [p5;p95], P) | seeds0-4 Δ | LOFO Δ (approx thr) | heavy Δ |
|---|---|---|---|---|
| all classes (v2) | -0.0005 [-0.0175;+0.0147] P0.50 | — | — | kurg -0.024/-0.006 (fail) |
| **arch only** | **+0.0097 (boot +0.0085 [+0.0012;+0.0147] P0.966)** | +0.0052/+0.0076/+0.0101/+0.0093/+0.0077 (5/5, P 0.89-0.98) | +0.0086 P0.967 | 0 |
| finds only | +0.0028 P0.869 | 5/5 >0 | **-0.0119** P0.22 | 0 | (3 families only — unreliable)
| honest select per class / group | +0.0037 P0.685 / +0.0056 P0.888 | — | — | gor -0.0058 (fail) |
| kurgan subclass (sub.py, AUC 0.597) | all variants -0.006…-0.030 | — | — | fail |
Arch gain: +24 TP, -506 FP (550 FP removed in Елизаветинское, no arch GT); without that family Δ still +0.0070, without Каменные +0.0070.
Files: nested2.py/nested3.py, work/nested_v2.*, work/v3/{res_*,counts_*}.json/pkl, sub.py, work/sub.json.
