# RESULT W4 C1 / GA / FP1 re-gate under PREREG d366a45 + AMEND1 c60e50f (02.11.2026 14:31 MSK)

Rule: prod τ refit (FP 0.5, metric._group_cuts midpoint). Identity sanity: **0 / 50** τ changed (old C1F rule: 50 / 50).
Base replay 0.345439 (PH 0.345439 OK); gate base 0.3337867 OK. Paired site-family bootstrap 2000, seed 0.

| variant | Δall | P | bar | Δnew34 | Δnolidar | per-class Δ (floors) | gate-side nested Δ | PASS |
|---|---|---|---|---|---|---|---|---|
| C1 | +0.016202 | 1.0 | 0.9 (reg.); ≥0.985714 too | +0.000747 | +0.033057 | kp +0.00935, kt +0.00175, gor +0.05092, arkh +0.05386, fort +0.00046 (all OK: True) | see below | **PASS** |
| GA | +0.013421 | 1.0 | 0.985714 | +0.000000 | +0.019520 | gor +0.05357, arkh +0.05386 (all OK: True) | see below | **PASS** |
| FP1 | +0.008581 | 0.999 | 0.985714 (W1 0.98333) | +0.000896 | +0.018988 | kp +0.01148, kt +0.00200, arkh +0.04740, fort -0.00007 (all OK: True) | see below | **PASS** |

Nested choices (d0, s) per fold: C1 [(0.25, 0.75), (0.25, 0.75), (0.0, 0.75), (0.25, 0.75), (0.0, 0.75), (0.25, 0.75), (0.25, 0.75), (0.25, 0.75), (0.25, 0.75), (0.0, 0.75)] ; GA [(0.25, 0.75), (0.25, 0.75), (0.0, 0.75), (0.25, 0.75), (0.0, 0.75), (0.25, 0.75), (0.25, 0.75), (0.25, 0.75), (0.25, 0.75), (0.25, 0.75)] ; FP1 fixed d0 0.5 edge 0.5.

Gate-side nested Δ (frozen): C1 +0.015165, GA +0.012177, FP1 +0.007715 (= W1's FP1 gate-side +0.007715). FP1 replay = W1's numbers exactly (Δ +0.008581, new34 +0.000896, nolidar +0.018988): the effect is real in the production code path ([post.w4_sat], 07fa43c, with scene_drel on sp_dfine).

Notes: GA new34 Δ = 0 exactly (gor/arkh have no effect on new34) → meets ≥ 0. GA classes are post hoc (disclosed). GAFP dropped (AMEND1). Best single PASS by Δall = **C1**. v4 NOT built (coord wrap-up order).
