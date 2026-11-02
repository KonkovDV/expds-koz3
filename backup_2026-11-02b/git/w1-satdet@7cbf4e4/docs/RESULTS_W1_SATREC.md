# W1-SATREC results (02.11.2026 MSK, bot-W1-satrec)

PREREG: `docs/PREREG_W1_SATREC.md` (140839a, committed before measurement). Context: v3_on b447649, production replay PH
(tools/v3prod on bots/w1-cfdeploy 9ea621d) + scene_drel annotation. Code: `tools/satrec/` (s1 gate-side nested selection,
s2 production replay per variant, s3 gate + audit). Flags (default OFF): `[post].sat_fusion`, `[post].sat_center_prior`
(`expds/sat_recall.py`, hooks in `expds/watchdog.finish` before `frame_prior`, both main and fallback chains).

## Sanity
- Gate base (stack, honest-10) = 0.3337867 = V3P (exact). Production replay base = 0.3454388 = PH v3 0.345439 (exact).
- Table sp geoms UTM -> 3857: 104326/104330 member centroids inside their region's sp_or scenes.
- Own global greedy TP == official build_curves TP for all 6 replays.

## Gate (production replay, honest-10 nested; Bonferroni m=5 -> P >= 0.98)

| var | all | Δall | P | Δnew34 | Δnolidar | kp | kt | gor | arkh | fort | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| base | 0.34544 | | | | | | | | | | |
| R1 sp_dfine single τ (0.65 all folds) | 0.35539 | +0.00995 | 0.997 | −0.00232 | +0.02780 | +0.0072 | +0.0041 | +0.0307 | +0.0236 | +0.0005 | FAIL (new34) |
| R2 equal fusion, per-class τ | 0.34782 | +0.00238 | 0.676 | −0.00221 | −0.00921 | +0.0040 | −0.0100 | +0.0204 | +0.0014 | −0.0001 | FAIL |
| R3 fusion + single τ (0.30 all folds) | 0.34083 | −0.00461 | 0.208 | −0.01844 | +0.01645 | +0.0058 | +0.0025 | −0.0692 | +0.0367 | −0.0032 | FAIL |
| C1 centre prior on sp_dfine | 0.35965 | +0.01422 | 0.9995 | −0.00110 | +0.03306 | +0.0096 | −0.0023 | +0.0426 | +0.0539 | +0.0005 | FAIL (new34, P4 kt) |
| C2 centre prior on sp_fuse | 0.35753 | +0.01209 | 0.983 | −0.00368 | +0.02485 | +0.0148 | −0.0033 | +0.0169 | +0.0574 | +0.0005 | FAIL (new34, P4 kt) |

Raw P are paired family bootstrap p_gt0 (2000). Nested centre-prior choice (training-fold gate score, ties -> smaller s, larger d0):
C1 (d0, s) = (0.25, 0.75) in folds 0,1,3,5,6,7,8 and (0.0, 0.75) in folds 2,4,9; C2 (0.0, 0.75) in 9 folds, (0.25, 0.75) in fold 5.
s = 0.75 is the grid edge (PREREG grid).

## Satellite recall-loss audit (base = v3_on, sat GT = markup sp_or, 1606 objects)
Coverage = GT with ≥1 candidate passing the metric's hit test (any-hit, no greedy). Per fold (folds 0..9):

| stage | total | per fold |
|---|---|---|
| sat GT | 1606 | 663 84 86 116 14 44 225 59 254 61 |
| L0 any detector (r18 ∪ f2 ∪ D-FINE raw) | 1268 | 531 22 63 87 13 42 182 55 231 42 |
| L0 D-FINE raw ellipses (conf ≥ 0.2) | 1188 | 501 17 58 80 13 40 179 53 206 41 |
| L0 r18+f2 (table, conf ≥ 0.05) | 879 | 374 7 20 46 8 34 138 49 182 21 |
| L1 D-FINE circles in (cluster_circles 4/4 + finalize; no floors on sp_dfine) | 1040 | 451 1 47 69 10 35 168 51 178 30 |
| L2 after thresholds (honest per-class τ) | 361 | 166 0 6 31 0 23 49 22 51 13 |
| L3 rerank / merge / multipole / finds / twins / micro / finalize | 361 | unchanged at every step |
| L5s matched (global greedy) by sat outputs | 355 | 166 0 4 31 0 22 48 21 51 12 |

Loss attribution: never detected 338 (21 %); r18/f2 dropped in v3 (unique beyond D-FINE) 80; D-FINE clustering 148;
**thresholds 679 (dominant, 65 % of D-FINE candidates)**; post-threshold stages 0; matching 6.
Counterfactual L1′ (r18+f2 at gate-honest τ, held fold): 178. Sat precision/recall (final): base 0.223 / 0.221;
R1 0.307 / 0.229; R2 0.220 / 0.225; R3 0.224 / 0.238; C1 0.245 / 0.267; C2 0.245 / 0.281.
By class at L2 vs GT: kp 263/1087, kt 39/207, arkh 53/210, gor 6/65, fort 0/37 (D-FINE circles cover only 2 fort GT).

Details: `docs/results_w1_satrec/{s3_gate.json,s3_audit.json,prep_summary.json}`.

## Notes / deviations
- Fusion members come from the table (r18/f2 rows, conf ≥ 0.05) and the D-FINE OOF circles; in the replay the 12 m σ rule
  is not applied to f2 because f2 is consumed by the fusion. In production the hook sits before `frame_prior` in `finish`.
- 11772 of 85069 D-FINE circles have no sp_or scene box (frame_boxes.json) -> centre prior leaves them unchanged.
