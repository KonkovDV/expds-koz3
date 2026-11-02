# LIKT-FROZEN re-gate PREREG (W3, 02.11 MSK). Committed before any measurement of the frozen variants.
Task from bot 1, 13:53: audit after the Q0/C1F lesson.
## Disclosure: this rule is post hoc
The rule "freeze out-of-layer thresholds at their baseline values" was adopted **after** Q0, and after the LIKT result (0b35b80) was known. It is not part of the original LI_KURG_TAU_PREREG.
## Audit findings that trigger this re-gate (runs/audit_LIKT.json/.log; diagnostic only, no gate run)
- **LIKT A.** D.honest re-swept all kurgan groups jointly, which changed 27 out-of-layer (non-li) thresholds across all 10 folds. On new34, non-li kurgan TP fell: intact 12→0, damaged 4→3.
- **LIKT C.** 3 out-of-layer changes, in folds 3, 4 and 9.
- **KDUAL (kdual.py).** Uses the baseline V3P per-fold taus for every existing row by construction (base_counts/adds use taus[k]); only tau_dual is nested. No out-of-layer re-tuning, so KDUAL is not re-gated.
## Method
Variants A, B1, B2 and C are exactly as in prereg/LI_KURG_TAU_PREREG.md (score and group definitions unchanged). Base is V3P (0.333787), honest-10 family folds.
Per outer fold k:
- Every group of every class that is not in-layer keeps the V3P baseline threshold tB[k] (from D.honest on V3P), frozen.
- In-layer groups are kurgan-class groups whose base name (before "|") is in {li, li_dev, li_mosaic}, including the C "|x" groups. Their thresholds are swept on the train regions of fold k, holding the frozen groups' TP/N constant.
  - Same cuts (M._group_cuts) and coordinate descent from "none" and "all" as D.sweep_fixed (≤20 rounds), maximising class F1 on train.
- An in-layer group with no train rows is dropped (unseen = drop, as in D.honest).
- Held-out counts use these per-fold taus.
Sanity: running the frozen sweep on unchanged V3P curves must reproduce V3P exactly (0.333787).
## Gate (each variant vs V3P; paired family bootstrap 2000)
Bonferroni m = 4 (re-gates A, B1, B2, C): P ≥ 1 − 0.1/4 = 0.975.
Accept if all hold:
- Δ_all ≥ +0.003;
- new34 Δ ≥ 0, nolidar Δ ≥ 0;
- ΔF1 damaged ≥ 0, intact ≥ 0, fort ≥ 0;
- floors: kp/kt/gor −0.002, others −0.005.
Flag li_kurg_tau_frozen, default off. Report Δ, P, new34, floors.
