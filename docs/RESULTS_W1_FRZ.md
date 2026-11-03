# RESULTS W1-FRZ (PREREG docs/PREREG_W1_FRZ.md 1532475), 02.11.2026 MSK

v3_on b447649 production replay PH, honest-10 nested. Every non-sp_dfine τ is frozen at the base fold τ, and only sp_dfine τ is re-fit.
Sanity checks:
- Base PH = 0.345439 (ok, ±5e-4).
- Gate-side base = 0.3337867 (ok).
- The frozen re-fit at identity reproduces the base sp_dfine τ: 0 mismatches.
- The freeze holds in every chosen fold (asserted).

Family m=3, P ≥ 0.98333. Bootstrap: paired family, 2000 resamples, seed 0, on prod-replay counts.

| Variant | Δall | P | p5..p95 | Δnew34 | Δnolidar | P4 (kp / kt / gor / arkh / fort) | Verdict |
|---|---|---|---|---|---|---|---|
| C1F | +0.01592 | 1.000 | +0.0080..+0.0249 | 0.0000 | +0.0285 | +0.0088 / +0.0003 / +0.0535 / +0.0519 / +0.0004 | **PASS** |
| A1F | +0.01463 | 1.000 | +0.0069..+0.0232 | +0.0008 | +0.0213 | +0.0078 / **−0.0033** / +0.0627 / +0.0353 / +0.0005 | FAIL (kt floor −0.002) |
| P2F | +0.01491 | 1.000 | +0.0070..+0.0241 | 0.0000 | +0.0211 | 0 / 0 / +0.0733 / +0.0392 / 0 | **PASS** |

object_poly and finds_points: 0 in all variants.

Gate-side nested Δ (stack): C1F +0.01487, A1F +0.01378, P2F +0.01362.

The originals (thresholds not frozen) failed new34: C1 −0.0011, A1 −0.0017, P2 −0.0027 (W4 Q0, 7c33027). With the freeze, new34 is ≥ 0 in all three.

## Chosen per fold (f0..f9)
- **C1F** (one s for all D-FINE classes):
  - (d0, s) = (0, .75), (.25, .75), (0, .75), (.25, .75), (0, .5), (.25, .75), (0, .5), (.25, .5), (0, .75), (.25, .5).
- **A1F** (d0, s_o, s_kt):
  - f0–f4: (0, .75, .5), (.25, .75, .5), (0, .75, .75), (.25, 1.5, .5), (0, 1.0, .5)
  - f5–f9: (.25, .75, .5), (0, .5, .5), (0, 1.0, .5), (0, 1.0, .25), (.25, 1.5, .75)
- **P2F** (d0, s_gor, s_arkh):
  - f0–f4: (0, .75, .5), (0, 1.5, 1.25), (0, 1.25, 1.0), (0, 1.5, 1.25), (0, 1.5, 1.25)
  - f5–f9: (.25, 1.5, 1.5), (0, 1.5, .5), (0, 1.0, 1.25), (0, 1.0, 1.25), (.25, 1.5, 1.5)

The sp_dfine τ for each fold are in results_w1_frz/prep_summary.json. Note that s = 1.5 (the grid edge) is chosen often in P2F.

## Notes
- Disclosure (as in the PREREG): the freeze rule was motivated by W4 Q0, which looked at new34 labels. new34 is therefore not a blind check here.
- There is no single production config. The parameters differ per fold, so a production config would need one full-data selection (not done here).
- The prior is applied at the production location via a replay-only wrapper. The original C1/A1/P2 replays applied it at feature build time.
- Scripts are in tools/satq: frz_prep.py, frz_replay.py, frz_score.py, run_frz.sh. Outputs are on the box at /workspace/koz3/w1/satq/frz (not in git).
