> **SUPERSEDED (02.11.2026 MSK)** — the sp_dfine τ refit here used plain F1 with τ = score instead of the metric/prod rule (FP 0.5, midpoint cuts); with no prior it changes 50/50 base τ. Re-run under docs/bots/2026-11-02/bot-W4/c1fx_regate/PREREG_AMEND1.md. This FAIL is not a result.

# W4 independent re-gate of W1 C1F — RESULT: FAIL (02.11 14:12 MSK)

PREREG e6b5a31 (committed before measuring). Base = v3_on b447649. Code = b447649 + W4 port (98cdf56 + v4_on mirror commit):
expds/sat_recall.py (af6b827 as is), watchdog finish/safe_finish flag hook + sat_center_prior before frame_prior,
pipeline scene_drel under the flag incl. **fix for sp_dfine** (D-FINE features are created after optical_feats, so in
both b447649 and af6b827 they never got scene_drel → the C1F prior would be a silent no-op in the archive).
Flags off in v3_on.toml / v4_on.toml; pytest 474 passed, 43 skipped (463 + 11 new).

Score scale: archive config weights = models/dfine_f0 (dfine_final only a reserve); v3_on thresholds were fitted on the
fold-model OOF on the same scale → no recalibration needed.

Gate-side nested (non-sp_dfine τ frozen at taus_V3P): base 0.3337867 (sanity exact); C1 0.344729 (Δ +0.01094); P2 0.347622 (Δ +0.01384).

Decision = production replay through the ported code, [post.w1_c1f] enabled with per-fold nested values (honest-10,
compute_metrics/s3 gate code, paired bootstrap site_family 2000):

| variant | Δall | P | new34 Δ (P) | nolidar Δ | class floors |
|---|---|---|---|---|---|
| base (flags off) | 0.345439 = PH sanity OK | | | | |
| **C1 (primary, P≥0.9)** | **+0.01224** | 0.999 | **−0.00079** (0.22) | +0.01146 | **kurgany_tselye −0.00333 < −0.002** |
| P2 (comparison) | +0.01529 | 0.999 | −0.00159 (0.12) | +0.00842 | kurgany_tselye −0.0059 |
| C1F_frozen (diagnostic, W1 full-data τ, in-sample) | +0.02719 | 1.0 | +0.0009 | +0.0590 | all OK |

C1 FAILS two PREREG criteria (new34 ≥ 0; kurgany_tselye floor). The frozen full-data values look much better (+0.027)
only because their sp_dfine τ were fitted on the same regions (optimistic; not a gate). v4 archive NOT built, Kaggle NOT run.

Files: c1f_gate.json (decision), c1f_gate_frozen_diag.json, prep_summary.json (per-fold d0/s/τ), scripts/, logs/.
