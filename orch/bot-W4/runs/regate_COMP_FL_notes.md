# COMP_FL independent re-gate (W4, 02.11 MSK)
PREREG 5aa57b9 PREREG_COMP_FL.md; code coord-comp 5dedde3 sat_frame_gate (t2_cls) verbatim, injected after C1F sat_center_prior, before thresholds.
Context: honest-10 v4 prod replay (C1 nested C1F, base == v4build out_C1 sha 3d286d22…, all 0.361641); τ frozen (0/50).
Result: Δall +0.000119 (P 0.5285, p5 −0.0023), new34 +0.000558, nolidar −0.002827, floor kp −0.00271 (<−0.002 FAIL).
Decision: FAIL; |Δ|<0.003 → lever CLOSED. No LB.
Artifacts: runs/regate_COMP_FL.json; comp/fl_replay.py, comp/fl_score.py; w4geo/comp/rep_{C1,FL}.geojson, gate_C1_FL.json
