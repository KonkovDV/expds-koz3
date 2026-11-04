# li_r50c5 on V3P, v3_on prod context: P-W2-9 amendment 2 (01.11.2026 22:15 MSK)

**Verdict: FAIL** (Δ +0.002391 < 0.003; P 0.6565 < 0.95; class floors broken: tselye −0.0077, gorodishcha −0.0151, fort −0.0052; new34 −0.0132; nolidar −0.0019).
The r50c5 line stays closed for the archive. No archive-fit work was done, because that step only runs on PASS.

## What was measured
- **A = V3P_ctx.** These are the V3P rows that survive the early floors of the v3_on archive. V3P itself is base3 + C19-etgr + D-FINE sp_dfine, as in W4 v3_gate_lib.
  - A row survives when conf ≥ output_floors(cfg_v3on, threshold_group)[class].
  - The config is configs/default.toml from w4geo/dist/koz3_v3on_b505656_6d4764f8_w6dfd2e07.zip (sha256 6d4764f8…).
  - v3_on has no li_floor (no w5_cloudfeat), and its prior and calibration are off. So each floor equals the group threshold. The floors are in v3on_floors.json.
- **B = A + li_r50c5.** These are the r50c5 OOF rows, 5/5 folds, 22119 in total; 5072 survive the 0.15/0.20 floor, since the li_r50c5 section does not exist.
- **Engine.** d3_gate TableEngine with honest-10 family folds (group thresholds re-swept, unseen=drop) and a paired bootstrap of 2000 by site_family, seed 0.

| | value |
|---|---|
| S0: V3P on this engine | 0.3337867244 (d = +4e-11 vs 0.3337867244, OK) |
| A = V3P_ctx | 0.345852 (stack rows kept 47,382 / 1,281,976; extras 3,563 / 86,647) |
| **B (decisive)** | **0.348243, Δ +0.002391, P 0.6565, CI90 [−0.0069, +0.0093]** |
| per class ΔF1 | povr +0.0216, tselye −0.0077, gor −0.0151, fort −0.0052, others 0 |
| new34 / nolidar | −0.013230 / −0.001876 |
| report only: B vs V3P without ctx | Δ +0.014456, P 0.9895 (the gap comes mostly from the context, which is not part of r50c5) |
| report only: li_r50c5 floor 0.05 | same as decisive (the sweep picks thresholds ≥ 0.15 in every fold) |
| report only: fixed archive thresholds, no sweep | A 0.402532 (in-sample), B Δ −0.009379, P 0.10, tselye −0.0504 |

## How to rerun
`nice -n 10 /workspace/koz3/w4venv/bin/python gate_r50c5_ctx.py 2000 out.json` takes about 40 s with warm caches and peaks at about 2.5 GB.
It expects `v3on_src/configs/default.toml` next to the script. Extract it from the archive zip above; the box copy is in /workspace/koz3/w2/r50c5_gate/.
OOF check: `validate_oof.py` → oof_validation.json (5/5 folds, region_fold f01 = f234, families not split, 4492 + 17627 rows).

## Where things are
- Box: /workspace/koz3/w2/r50c5_gate/ (script, log, json, v3on_src).
- OOF: /workspace/koz3/w2/oof/r50c5/{elenakonkova_…-f01, dmitrijkv_…-f234}.
- PREREG: orch_bot-W2/PREREG_W2_cnx.md, P-W2-9 amendment 2 (commit 29e6aed, made before the measurement).

Caveats (this is an approximation): the stack is used through table.pkl. In the stack, prod micro_circles has not been applied to the kurgans; in r50c5 it already has (0.5 m circles). merge_within_markup does not merge within one LAS source. twins applies only to ae/sp, and frame_prior only to sp_*.
