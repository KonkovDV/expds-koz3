# W4 independent re-gate of W1 C1F (main) / P2F (comparison) — PREREG
bot-W4, 02.11.2026, written before any measurement of this re-gate. Base = v3_on b447649 (archive koz3_v3on_b447649_…zip,
sha256 1406c01a…). Source of the flags: bots/w1-satq af6b827 (docs/W1_C1F_HANDOFF.md).

## Pre-measurement findings (code reading only)
1. **Score scale.** The archive runs `[sp_dfine] weights = ["models/dfine_f0"]` (zip contains models/dfine_f0, not dfine_final).
   The gate's D-FINE inputs are the f0..f4 fold-model OOF (w4geo/spor_oof), the same family on which the v3_on sp_dfine
   thresholds were fitted (W1: identity sweep reproduces them exactly). C1F τ are therefore on the scale the archive produces;
   **no recalibration**. (dfine_final is a reserve model only and is not used.)
2. **scene_drel under the flag — BUG in af6b827.** In production, D-FINE features are created in
   `pipeline._sp_dfine_phase`, *after* `optical_feats`, the only place that calls `_with_scene_drel`. Neither b447649 nor
   af6b827 annotates sp_dfine features, so in the archive the C1F prior would be a **no-op** and only its (lower) sp_dfine
   thresholds would act. W1's replay annotated every feature from frame_boxes.json, which hides this.
   Fix (part of the C1F port, flag-gated, off = byte-identical behaviour): in `_sp_dfine_phase`, when `[post]
   sat_center_prior` or an enabled `w1_c1f`/`w1_p2f` is present, `found = _with_scene_drel(reg, cfg, found)` (same
   map_footprint boxes, markup sp_or, EPSG:3857). Unit test for it.

## Port
v4 code = b447649 + minimal port of af6b827: `expds/sat_recall.py` (as is), `watchdog.finish` hook (resolve_final_flag at
entry; `sat_center_prior` before `frame_prior`), the scene_drel condition in `optical_feats`, the fix above, the two config
tables (both `enabled = false` in v3_on.toml), tests. Nothing else from the W1 branch.

## Gate (honest-10, same compute_metrics.py via expds.metric, folds t_taus.json)
- Selection per fold fi on training regions `every − fold_fi` (nested); **all non-sp_dfine thresholds frozen at taus_V3P[fi]** (Q0 rule).
- C1 grid: d0 ∈ {0, .25, .5, .75} × s ∈ {.25, .5, .75} (same s for all 5 D-FINE classes) + identity (s = 0).
  For each (d0, s): prior applied to sp_dfine OOF conf (scene_drel from frame_boxes.json = production footprints), curves
  rebuilt (gate greedy matching as in W1 s1), per class the sp_dfine τ = argmax of that class's training F1 over the
  sp_dfine cut points (exact 1-D, highest τ among ties; "none kept" allowed → τ = 1.01). (d0, s) = argmax training score;
  ties → smaller s, then larger d0.
- P2 grid (comparison only): d0 ∈ {0, .25, .5}, s_gor, s_arkh ∈ {0, .25, …, 1.5}, other classes s = 0; same τ rule.
- Held-out decision: **production replay** of the ported archive code (copy of x_v3on b447649 + port) per fold with
  `[post.w1_c1f]` enabled, its d0 / s_by_class / thresholds_sp_dfine = that fold's nested choice, other τ = taus_V3P[fi];
  feature assembly = W1 s2_replay (table.pkl + D-FINE OOF extras, scene_drel from frame_boxes.json). Scored with
  W1 s3 gate code (compute_metrics via expds.metric.build_curves "global"; paired family bootstrap 2000, seed 0).
- Sanity: ported code, flags off → base replay = 0.345439 (± 5e-4); gate base = 0.3337867.
- Report both: frozen (primary) and gate-side nested score.

## Criteria (C1F only; P2F is reported, never decides)
PASS iff all: Δall ≥ +0.003; P(Δ>0) ≥ 0.90 (C1F is W1's pre-chosen single primary); per-class F1 floors
kp/kt/gor ≥ −0.002, others ≥ −0.005; Δnew34 ≥ 0; Δnolidar ≥ 0.
On PASS: v4 archive = b447649 + port with `[post.w1_c1f] enabled = true` and W1's full-data values
(d0 0.25, s 0.75 all, τ 0.384082/0.426006/0.56171/0.281948/0.525625), DOC-1 guard, check_archive_clean, WEIGHTS_MANIFEST,
one v4 zip, sha256, Kaggle parity (C_default, A_regex2025_preloaded, B_no_regex, D_new_env521, NEG) via pool_kaggle.sh.
