# PREREG AMEND1 — W4 C1F / GA / FP1 (02.11.2026 ~14:35 MSK, bot5-W4; committed BEFORE the re-measurement)

Amends PREREG.md (d366a45) of this folder and supersedes the W4 C1F re-gate (PREREG e6b5a31, RESULT 5d0dd4c).

## Why
W1 (bots/w1-satq 3929737, docs/W1_FRZ_DISCREPANCY.md) reported that W4's sp_dfine τ refit (c1f_prep.sweep) used plain F1
(FP weight 1) with τ = a score itself, while the metric / production threshold fit uses F1 with FP weight 0.5 (metric._f1)
and τ at the midpoint between scores (metric._group_cuts). **Independently confirmed by W4** (scripts/ident.py,
ident_check.json): with no prior, over 10 folds × 5 D-FINE classes the C1F re-gate rule changes **50 / 50** base τ;
an intermediate rule (FP 0.5, τ = score) also changes 50 / 50; the prod rule (FP 0.5, _group_cuts midpoint, first argmax,
round 6) changes **0 / 50**.

## Disclosure of what was seen before this amendment
The d366a45 run with the intermediate rule (FP 0.5, τ = score) was fully measured: production-replay GA Δ +0.0134
(P 1.0), FP1 Δ +0.0086 (P 0.999) — both nominally passing; one earlier gate-side run with the plain-F1 rule
(GA +0.0107, FP1 −0.0014). None of these are results; they are superseded by this amendment.

## Changes
1. Freeze-rule τ refit of a touched sp_dfine class = the prod rule above (all other τ as in PREREG.md).
2. **Sanity gate (blocking):** no prior → 0 sp_dfine τ changes vs base (all 50 reproduced); asserted in prep before
   any variant; base replay 0.345439 and gate base 0.3337867 still asserted.
3. Variants, each gated separately (production replay, same scoring, same flag [post.w4_sat]):
   - **C1 (C1F, all 5 D-FINE classes)**: as PREREG e6b5a31 C1 — one (d0, s) for all sp_dfine classes, nested over
     d0 ∈ {0, .25, .5, .75} × s ∈ {.25, .5, .75} + identity (ties → smaller s, then larger d0); touched τ: all 5.
     Bar: P ≥ 0.9 (its registered bar, e6b5a31); also reported vs 0.985714.
   - **GA (C1F-gor/arkh, post hoc classes — disclosed in PREREG.md)**: unchanged; P ≥ 0.985714 (m = 7).
   - **FP1 (W1 as is)**: unchanged; P ≥ 0.985714 (W1's 0.98333 reported).
   Other criteria unchanged: Δall ≥ +0.003, Δnew34 ≥ 0, Δnolidar ≥ 0, floors kp / kt / gor −0.002, others −0.005.
4. **GAFP (combination) dropped** — not gated (coord: W1 preliminary shows it is not additive, kt −0.0035).
5. v4 = the single PASS with the largest replay Δall (none → nothing built); full-data values by the same rules on all
   training regions.
