# Handoff results 01.11.2026 ~21:45 MSK (bot taking over from the out-of-quota orchestrator)

## Kernels
| Kernel | Status | Result | Verdict |
|---|---|---|---|
| elenakonkova/koz3-w4-large-gate (run 21:09 MSK) | COMPLETE | honest-10 vs A 0.314615: baseline Δ0, area_tau Δ0 (inert at τ=1.01), fixc_nested (+698 D-FINE ae gor OOF) 0.314499 **Δ−0.00012**, P 0.0; vs B′ abs −0.0013 | **FAIL** |
| elenakonkova/koz3-w1-ae-dfine-gate (rerun 21:22 MSK) | ERROR (script FAILED, flag_dead) | 1-region smoke (015_Демидовка): OFF/A+B/A+B+C all F1 0.0; A+B+C adds 4 ae dets (3 kurg_tselye, 1 fort), all FP. Not an honest-10 gate; the honest-10 part of this lever = large-gate fixc_nested above | **FAIL** |
| dmitrijkv/koz3-w1-ae-dfine-gate | ERROR | no access to elenakonkova dataset (known) | — |
| dmitrijkv/koz3-w1-lidar-aug-f2 (c7-sc R-augx, fold 2) | COMPLETE (2.8 h) | Li-only thr0.3 **0.3432** vs c3-sc ref same fold/regions **0.4006** (−0.057); best-thr in-sample 0.409 vs 0.455; raw 0.160 vs 0.149. Early-stop rule (fold < ref−0.01) → STOP. 3-fold mean f0,f1,f2 0.3764 vs ref 0.3663 (+0.010) | **fold FAIL** |
| elenakonkova/koz3-lidar-aug-f0 (v5, folds 3,4) | RUNNING since 15:20 MSK, deadline_h 11 → ≤ ~02:20 MSK 02.11 | not touched | pending |
| dmitrijkv/koz3-w1-sat2-check (combo pack check) | RUNNING | not touched | pending |

## v2+sat2(+FR1h) vs v3_on — same engine (PREREG docs/PREREG_h2h_sat2_v3on.md @879453b)
Engine: W4 v3_gate_lib (TableEngine, honest-10, family boot 2000). B0 = v2clean 0.313967.
| Variant | all | new34 | nolidar |
|---|---|---|---|
| B0 v2clean | 0.313967 | 0.296451 | 0.143623 |
| **V3P = v3_on** | **0.333787** | 0.304855 | 0.218699 |
| S2 = v2 + sat2 REPLACE | 0.323554 | 0.303234 | 0.185100 |
| U = V3P + sat2 | 0.335644 | 0.303806 | 0.225854 |
- S2 vs B0 +0.0096 P0.897 (= W1 sat2 gate scale1.0 exactly); V3P vs B0 +0.0198 P0.976.
- **S2 vs V3P −0.0102, P(S2>V3P) 0.053** → v3_on line is higher.
- U vs V3P +0.0019, P0.941, new34 −0.0010 → **FAIL** (needs Δ≥0.003, P≥0.95). sat2 not added.
- Lines are not composable as-is: both replace the satellite sp_* kurgan rows (sp_dfine vs sp_det2).
- FR1h: dropped by the combo gate itself (sat2+FR1h 0.32563 < sat2 0.32645 in gate_v6c harness).
- Combo pack (bots/combo-sat2-fr1h 2722024, zip sha d85e9b2f…, ~1069 MiB, sat2 only) lives on the other bot's box + Drive; not on this box.

**Decision: final stays v3_on (koz3_v3on_b505656, sha ebc8e4ba…, check PASS); fallback v2clean 545ef32a. v2+sat2 = second fallback only.**
