# W1 FRZ C1F vs W4 C1F re-gate: cause of the discrepancy (02.11, 14:19 MSK)

W1 FRZ (5cc5ed7): C1F prod replay Δ +0.0159, new34 0, kt +0.0003, PASS.
W4 re-gate (bots/w4-v3on 5d0dd4c, PREREG e6b5a31): C1 prod replay Δ +0.0122, new34 −0.0008, kt −0.0033, FAIL.

## What is the same (checked)
- Folds: honest-10, both from orch/bot-W1/work/v3prod/t_taus.json.
- Inputs: the same D-FINE OOF (85069 circles; 11772 have no drel) and the same drel (bot-10 frame_boxes.json via
  post.annotate_scene_drel in both replays).
- Base: the same base, with gate-side base 0.3337867 and prod base 0.345439 in both.
- Non-sp_dfine τ: per-fold base τ, identical (0 differences across all 10 folds).
- No full-data leak in W1: the sp_dfine τ differ by fold and are fit only on training regions. W4's +0.027
  "C1F_frozen" uses full-data τ; it is in-sample, and only a diagnostic.
- Candidates: the same 13-point (d0, s) grid, with the same tie rule (−s, then d0).
- Gate: the same s3 code and the same paired site_family bootstrap (2000 resamples).
- Where the prior is applied: in both replays, sat_center_prior runs on sp_dfine right before post.frame_prior.
  **Both replays add scene_drel to D-FINE themselves.** So neither replay saw the archive bug W4 found
  (D-FINE features never get scene_drel in the real pipeline). The gate numbers are not affected. But the af6b827
  production config would be a silent no-op in a real run. W4 fixes that in 98cdf56.

## Cause
The difference is **the sp_dfine τ refit rule inside W4's nested selection** (W4 c1f_prep.py `sweep`):
- W4's `sweep` uses plain F1 = 2tp/(2tp+fp+fn), which weights an FP as 1, and puts τ at the cut-point confidence.
- The competition metric and the production honest sweep that fitted the v3_on τ use something else:
  M._f1 weights an FP as 0.5 with eps 1e-6, and M._group_cuts puts τ at the midpoint between confidences.
  W1 frozen_taus uses that rule.
- Identity check at s = 0 (no prior): W1's rule reproduces all 50 base sp_dfine τ (0/50 mismatches). W4's rule
  changes all 50 (50/50 mismatches).

So W4's "C1" changes two things at once: it adds the prior and it swaps in a stricter τ objective. The stricter
objective gives systematically higher τ, which keeps fewer D-FINE kurgans and costs kurgan recall. That is where
W4's kt and new34 losses come from.

2×2 attribution, gate-side nested honest-10. Same curves, drel and candidates; the script is
/workspace/koz3/w1/satq/disc/diag3.py and the output is docs/w1_frz_disc/diag3.json:

| sp_dfine τ rule | (d0, s) selection objective | Δall | Δnew34 | Δkt |
|---|---|---|---|---|
| metric F1 (FP 0.5, midpoint) = production | pooled plain F1 (W1 FRZ) | +0.014869 (= FRZ) | 0.000000 | +0.000908 |
| metric F1 = production | official score_counts (W4) | +0.015165 | +0.000767 | +0.002376 |
| plain F1 (W4 sweep) | pooled plain F1 | +0.007033 | −0.000712 | −0.002124 |
| plain F1 (W4 sweep) | official score_counts (W4) | +0.010943 (= W4, exactly) | −0.000813 | −0.002724 |

Both extreme cells reproduce their originals exactly: W1 FRZ gate-side +0.014869, and W4's prep log
"C1 gate nested +0.010943" with the same per-fold choices.

**The τ rule flips the sign of new34 and kt. The selection objective does not.**

Secondary W1 defect, disclosed: the FRZ (d0, s) selection objective `sc` was pooled plain F1 (FP weight 1), not the
metric. With the metric objective, the C1F result is about the same or better (+0.0152, new34 +0.0008, kt +0.0024).

## Verdict on validity
- The W1 FRZ C1F gate number is a production-faithful *replay* with the production τ rule. On the gate question
  itself it is not invalidated.
- W4's FAIL comes from its non-production τ rule. It is not a property of C1F.
- What *was* not production-faithful is the af6b827 *config*: in the real pipeline the flag is a no-op because of
  the scene_drel-before-D-FINE bug. It needs W4's 98cdf56 fix before any expected-PH claim holds.
- I recommend W4 re-run its re-gate with the M._f1 / M._group_cuts τ rule. It should pass the identity check at
  s = 0, as FRZ does.

## W1-COMBO (PREREG 5691184)
It was **not** cancelled before measurement. It ran at 14:04–14:07 MSK, before both steering messages
(14:11 and 14:12). It used the same production τ rule as FRZ.

Result (docs/w1_combo/s3_gate_COMBO.json): **FAIL** on the kt floor.

| COMBO vs | Δall | P | Δnew34 | Δnolidar | kt |
|---|---|---|---|---|---|
| base | +0.011547 | 1.0 | +0.001953 | +0.013907 | **−0.00347** (floor −0.002) |
| C1F alone (info) | −0.004375 | 0.055 | +0.001953 | −0.014560 | |
| FP1 alone (info) | +0.002966 | 0.80 | +0.001057 | −0.005081 | |

The result is non-binding: bot 1 handed the FP1, gor/arkh and C1F+FP1 re-gates to W4. Nothing else was measured
after the stop.
