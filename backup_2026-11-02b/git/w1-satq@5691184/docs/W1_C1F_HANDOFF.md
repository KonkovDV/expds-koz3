# W1 → W4 handoff: C1F (main) / P2F (comparison) production flags

bot-W1, 02.11.2026, branch `bots/w1-satq`. Gate evidence: PREREG FRZ `1532475`, results `5cc5ed7`
(`docs/RESULTS_W1_FRZ.md`). Nested FRZ prod-replay gate (honest-10, m = 3, P ≥ 0.98333):
C1F PASS Δ +0.01592, P 1.000, Δnew34 0, Δnolidar +0.0285, every class ≥ 0. P2F PASS Δ +0.01491, P 1.000.

## What the flags do
Scene-centre prior on the D-FINE group `sp_dfine`, applied before thresholds (`watchdog.finish` / `safe_finish`, at the
`[post].sat_center_prior` step, i.e. before `post.frame_prior` and `apply_thresholds`):
`conf ← conf · max(0, 1 − s_c · clip((scene_drel − d0)/(1 − d0), 0, 1))`; objects without `scene_drel` (no covering
scene) are unchanged. The flag also replaces `[thresholds_by_markup.sp_dfine]` with its own values. **All other groups'
thresholds are untouched** (FRZ freeze). Code: `expds/sat_recall.py` `resolve_final_flag` / `sat_center_prior`;
`expds/pipeline.py` annotates `scene_drel` when `frame_prior`, `sat_center_prior` or a W1 flag is on (v3_on already has
`frame_prior` on, so nothing changes there).

## Flags (configs/v3_on.toml, [post]; both `enabled = false`)
| flag | d0 | s by class | sp_dfine thresholds (kp / kt / gor / arkh / fort) |
|---|---|---|---|
| `[post.w1_c1f]` **main** | 0.25 | all 5 D-FINE classes 0.75 | 0.384082 / 0.426006 / 0.56171 / 0.281948 / 0.525625 |
| `[post.w1_p2f]` comparison | 0.0 | gorodishcha 1.5, arkhitektury 1.25 | 0.555265 / 0.670285 / 0.383565 / 0.228759 / 0.869051 |
| (v3_on today) | — | — | 0.555265 / 0.670285 / 0.68847 / 0.373015 / 0.869051 |

To enable: set `enabled = true` in exactly one of the two tables; nothing else. Both on, or a flag together with a plain
`[post].sat_center_prior`, raises ValueError (in `safe_finish` the error is logged and cfg is used as is = flag off).

## How the values were chosen (once, all regions)
`tools/satq/full_prep.py` (same rule as the nested FRZ selection, train = all 132 D-FINE regions / 210 gate regions):
non-sp_dfine τ frozen at the full-data base sweep; per (d0, s) the sp_dfine τ per class = exact 1-D argmax of F1 over the
group cut points (first max, 6 decimals). C1F grid d0 ∈ {0, .25, .5, .75} × s ∈ {.25, .5, .75} + identity, tie → smaller s,
then larger d0. P2F d0 ∈ {0, .25, .5}, s_gor, s_arkh ∈ {0 … 1.5 step .25}. Check: the identity full-data sweep reproduces
the v3_on bot-9 sp_dfine thresholds exactly (0 mismatches), so the τ above are on the production score scale.
In-sample gate-side: base 0.339060 → C1F 0.353827 (+0.014767), P2F 0.357103 (+0.018043); C1F per class Δ:
kp +0.0066, kt +0.0003, gor +0.0660, arkh +0.0282, fort +0.0004, others 0 (all ≥ 0). Note the P2F choice sits on
the grid edge (s_gor = 1.5).

## Sensitivity (report only, selection unchanged)
s beyond 1.5 does not change either full-data choice: extended grid (s up to 2.0) best is still C1F (0.25, 0.75) and
P2F (0, 1.5, 1.25). In-sample C1-all scores at s 1.75 / 2.0: d0=0 0.34648 / 0.34510, d0=.25 0.34847 / 0.34953 (vs chosen 0.35383).
Nested honest-frozen, fixed points (no selection; base 0.33379): C1-all d0=.25 s 1.0 / 1.5 / 2.0 = 0.34683 / 0.34380 / 0.34027
(falls with s); P2 d0=0 s_gor=s_arkh 1.5 / 1.75 / 2.0 = 0.34601 / 0.35060 / 0.34915, (2.0, 1.5) 0.35057. P2 gains a little
above 1.5 nested, C1 does not. All numbers: `docs/results_w1_frz/full_summary.json` → `sensitivity`.

## Expected PH on the full-data config (production replay, in-sample)
Replay of v3_on b447649 (`/workspace/koz3/w1/satrec/x_v3on`, `configs/v3_on.toml`) with the same full-data τ in every fold:
| variant | PH all | new34 | nolidar | Δall (P) | Δnew34 | Δnolidar |
|---|---|---|---|---|---|---|
| B_full (base, full-data τ) | 0.415125 | 0.336004 | 0.294699 | — | — | — |
| **C1F_full** | **0.429082** | 0.337212 | 0.322448 | **+0.013957** (1.000) | +0.001208 | +0.027749 |
| P2F_full | 0.428820 | 0.336004 | 0.312664 | +0.013695 (1.000) | 0 | +0.017965 |

C1F_full per class Δ: kp +0.0064, kt +0.0011, gor +0.0617, arkh +0.0254, fort +0.0008, object_poly/finds 0 (all ≥ 0).
These are **in-sample** (τ of every group fitted on the same data); the honest estimate is the nested FRZ gate
(base PH 0.345439 → C1F +0.01592). Scores: `docs/results_w1_frz/s3_gate_C1F_full_P2F_full.json` (its `_sanity.ok = false`
only because the base here is in-sample, not the honest 0.345439). Output sha256 (box, not in git):
    a133d9bfcd5edcf5501bcba8c634958574b082d0902006ae4dea693ac77baa57  out_B_full.geojson
    8316ee4fe9451bbd8d5e18617a531b75a83cbf223b4526c7eeea08c7e66a4ca1  out_C1F_full.geojson
    6be5dfb983807c470e5fe5d252c6568a3c419b32a5d113eace90694e6af105e9  out_P2F_full.geojson

## Replay command
```
tools/satq/run_full.sh      # nice 10, ≤4 threads; = full_prep.py → frz_replay.py {B_full,C1F_full,P2F_full} … full.pkl → frz_score.py
```
Outputs go to `/workspace/koz3/w1/satq/frz/` (full.pkl, full_summary.json, out_*_full.geojson, s3_full.log).
Inputs: v3_gate_lib (orch/bot-W4/work), w5data stack/table, bot-10 frame_boxes.json, orch/bot-W1/work/v3prod/t_taus.json.

## What W4 needs for the v4 archive and Kaggle parity (no weights change)
* Code: `expds/sat_recall.py`, `expds/watchdog.py` (finish/safe_finish hook), `expds/pipeline.py` (scene_drel condition).
* Config: `configs/v3_on.toml` `[post.w1_c1f]` with `enabled = true` (or `[post.w1_p2f]` for the comparison build).
* Tests: `tests/test_sat_recall.py` (flags present/off, resolve, idempotent, conflicts, prior values, watchdog hook).
* Parity: production `scene_drel` comes from `map_footprint` of each sp_or raster; the gate used bot-10 frame_boxes.json
  (168/168 equal to production footprints per bot-7/work6/fr1h/boxes_cmp.out). For a Kaggle parity check compare
  sp_dfine confidences before thresholds against `conf · m` on a few SpOr regions, and the count of sp_dfine objects
  passing thresholds. The sp_dfine module's own pass-count diagnostic (`expds/sp_dfine.py`, before finish) sees the
  unresolved cfg, i.e. the old thresholds — log only, no effect on output.
* Same model: thresholds assume the D-FINE f0 score scale (as v3_on); not valid for dfine_final without recalibration.
