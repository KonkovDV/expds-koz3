# RESULT PREREG_COMP_KMASK — W3 (bot D) measurement, 02.11.2026 ~15:20 MSK
Spec: coord-A `5aa57b9` PREREG_COMP_KMASK.md (copy: orch/bot-W3/prereg/). Same frozen v4 baseline as GORLOC (out_C1, honest-10 0.361641). Measurement only; flag stays OFF.
Rule: drop sp_dfine kp/kt (markup sp_or in D-FINE regions) whose centroid lies within ρ·54 m (ground) of an emitted v4 gor sp_dfine point. All τ are frozen.

- GT check: sp_or kp+kt 1 294, inside sp_or gor **0** (matches A).
- **Oracle ceiling** (drop sp_dfine kp/kt inside GT sp_or gor polygons): 67 dropped, Δall **+0.00065** (P 1.0), nolidar +0.0016, new34 0, kp +0.0018, kt +0.0005.
  **Ceiling < +0.002 → per PREREG: do not gate.**
- Nested ρ (picks 0.75 ×9, 0.5 ×1): Δall +0.000035 (P 0.952), new34 0, nolidar +0.00009; kp +0.00006, kt +0.00009 (floors OK); LOFO min +0.00002.
- Fixed ρ (info only): 0.5 → 3 dropped, +0.000035; 0.75 → 4, +0.000044; 1.0 → 5, −0.000037.
- Why so small: v4 emits only 65 gor sp_dfine points, and almost no kept sp_dfine kurgans sit near them.
Files: `RESULT_PREREG_COMP_KMASK.json`, harness `work/gorloc/w3_comp.py`. No LB, no default.toml change.
