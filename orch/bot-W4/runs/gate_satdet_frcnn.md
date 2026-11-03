# W4 gate — satdet-frcnn (elenakonkova/koz3-w1-satdet-frcnn, COMPLETE) — 02.11.2026 ~16:31 MSK
**Verdict: FAIL (all variants).** Hold LB; v4 (out_C1) stays primary. CPU only, nice 10, no submit/push.

PREREG: `wt/coord-comp/docs/PREREG_W1_SATDET.md` (843d807), harness `wt/coord-comp/tools/satdet` (7cbf4e4).
PASS ⇔ Δall≥0.003 ∧ P≥0.975 (m=4 Bonf) ∧ Δnew34≥0 ∧ Δnolidar≥0 ∧ floors (kp/kt/gor −0.002, others −0.005).

## OOF (not empty)
11751 FRCNN ellipses (g0 3012 / g1 3840 / g2 1437 / g3 3462), 129 regions, one model per region (asserted) →
9297 sp_det2 circles after cluster_circles 4/4 + merge + finalize: kp 6419, kt 1319, gor 728, arkh 719, fort 112.
Train 9000 it/model, 16164 s total. (Files are 2.4–20 MB, not ~880 B.)

## A. PREREG family vs v3_on base (PH 0.3454388 reproduced; gate-side V3P asserted)
| var | Δall | P | Δnew34 | Δnolidar | floor fails | verdict |
|---|---|---|---|---|---|---|
| P1 | +0.013447 | 0.9960 | **−0.002747** | +0.020867 | none (gor +0.063, arkh +0.040) | FAIL (new34) |
| P2 | +0.013691 | 0.9965 | **−0.002747** | +0.021143 | none (gor +0.065, arkh +0.039) | FAIL (new34) |
| D1 | +0.005704 | 0.692 | +0.000237 | **−0.00767** | kt −0.0115, gor −0.0031 (kp −0.0018 ok; arkh +0.090) | FAIL |
| D2 | +0.008546 | 0.854 | **−0.000293** | +0.0059 | gor −0.0031 (arkh +0.090) | FAIL |
Gate-side nested: P1 0.344417, P2 0.344671, D1 0.334709, D2 0.339599 vs V3P 0.333787.
P1/P2 = track 2 (CPU centre prior), not the detector; they are subsumed by v4's C1 centre recipe anyway.

## B. W4 extra: vs v4 out_C1 (sha 3d286d22…, 0.361641) — NOT PREREG family
C1 frozen (per-fold centre d0,s + every C1 τ); only sp_det2 per-class τ refit nested on 9 training folds; bar P≥0.95.
ID (no extras) reproduces C1 byte-identically (sha 3d286d22…) and gate_nested 0.348952.
| var | Δall | P | Δnew34 | Δnolidar | floor fails | verdict |
|---|---|---|---|---|---|---|
| C1D1f (5 cls) | −0.00191 | 0.3945 | +0.00233 | −0.00021 | kp −0.0078, kt −0.0050, gor −0.0260 (arkh +0.0535) | FAIL |
| C1D2f (gor/fort/arkh) | +0.001442 | 0.584 | 0.0 | +0.002353 | gor −0.0260 (arkh +0.0535) | FAIL |
Gate-side nested: C1D1f 0.337706 (−0.0112), C1D2f 0.345965 (−0.0030) vs C1 0.348952.
Signal: FRCNN helps arkh (+0.05…+0.09 F1) but hurts gor; kurgan extras hurt. An arkh-only sp_det2 would be a new (post-hoc) variant — not gated.

## Paths / changes
- Pulled: `/workspace/koz3/w1/satdet/pulled/` (report + 5 geojson + kernel log).
- Local harness copy (shared tools untouched): `orch/bot-W4/comp/satdet/{t1_prep,t2_replay,t3_score}.py` — only paths changed:
  d3/cache→`v4build/gate/d3cache`, out→`comp/satdet/out`, x_v3on→`comp/satdet/x_v3on` (= `git archive b447649` of repo). run.sh.
- vs-C1 sidecar: `comp/satdet/c1det_{prep,replay,score}.py` (from comp/dfine2a), runc1.sh, outs `comp/satdet/outc1/`.
- Gate JSON: `out/s3_gate_P1_P2_D1_D2.json`, `outc1/c1det_gate_vs_C1.json`; sha in `out/sha.txt`, `outc1/sha.txt`.
