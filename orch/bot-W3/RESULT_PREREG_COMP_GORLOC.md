# RESULT PREREG_COMP_GORLOC — W3 (bot D) measurement, 02.11.2026 ~15:20 MSK
Spec: coord-A `5aa57b9` docs/bots/2026-11-02/coord-A/PREREG_COMP_GORLOC.md (copy: orch/bot-W3/prereg/). **Measurement only; final gate = W4. Not enabled anywhere.**
Clean-room: our code only (W4 c1fx counting, expds.metric, bot-5 gor_dense peaks + gor_v3 scene p). No EN code.

## Baseline (frozen)
- v4 zip `koz3_v4c1f_07795d2_51882efd_w6dfd2e07.zip` sha256 f0def982…202cef (verified).
- v4 OOF = W4 production replay `v4build/gate/out/out_C1.geojson` (C1F-all nested per fold; copy `work/gorloc/v4_out_C1.geojson` sha256 3d286d22…2ecc).
  honest-10 all **0.361641** (= W4 base 0.345439 + C1 Δ 0.016202 ✓), new34 0.315164, nolidar 0.254254.
- v4 emits 65 gor sp_dfine points in 36 regions (all in D-FINE regions; non-dreg sp_or gor = 0).
- Applied post-hoc on the final v4 output; **all existing τ frozen**; only L2 gets new group `gor_loc` (metric rule: FP 0.5, midpoint, first argmax, round 6) on training regions of each fold.
- Peaks: `peaks_fam10_v2prior.json`, `pu_edge+cp`, K=1 (folds identical to honest-10; train peaks = inner OOF). Held top-1 peak in sp_or GT: **31/49** (= bot-5).
- Point = 4 m circle (sp_dfine cluster_circles geometry, quad_segs 4). Peak must lie in an sp_or frame box.
- p_scene / gor_gate p: bot-5 gor_v3 `scene_p_oof_train` (5-family OOF; gor_gate is OFF in v4, so for L1 only "v4 emits gor sp_dfine" applies).
- Bootstrap: paired family bootstrap 2000, seed 0. Bar per PREREG: Δ ≥ +0.003, P ≥ 0.95 (Bonferroni k=2), new34 ≥ 0, nolidar ≥ 0, floors.

## Results (nested, honest-10)
| var | nested picks | Δall | P | p5 | Δnew34 | Δnolidar | gor TP/FP (base 37/101) | floors | LOFO (drop family) |
|---|---|---|---|---|---|---|---|---|---|
| L1 swap | m_min 0 (9/10), 0.5 (1) | +0.001325 | 0.602 | −0.0025 | 0 | +0.00256 | 38/100 | OK (gor +0.0088) | min 0.0; Kislovodsk out +0.0015 |
| L2 add+gate (g) | g 0.1 all folds; τ_gor_loc 0.149 (8), 0.934 (1), 0.146 (1) | +0.001799 | 0.6455 | −0.0064 | **−0.00133** | +0.00884 | 44/149 | OK (gor +0.012) | min −0.0024 = **Kislovodsk out −0.0024** |
| L2×m_min (secondary, g×m_min) | g 0.1, m_min 0.3/0.5 | +0.00388 | 0.811 | −0.0031 | **−0.00067** | +0.01118 | 44/132 | OK (gor +0.026) | min +0.00004 = Kislovodsk out |

Non-nested fixed grid (info only, not a pick): L1 m_min 0/0.3 +0.0013, 0.5 0; L2 g 0.1/0.2/0.3 → +0.0018/+0.0028/+0.0043.
Training selection always takes g = 0.1 (more points); held-out prefers g = 0.3, so the nested score is below the best fixed value.

## Reading (not a gate decision)
- No variant meets the registered bar together: L1 is too small (+1 TP, P 0.60); L2 has P 0.65 and new34 < 0; L2×m reaches Δ ≥ 0.003 but P 0.81 and new34 −0.0007.
- Gains are uneven across folds (L2: f2 +0.030, f8 +0.027; f6/f7 −0.006), and Kislovodsk carries part of the L2 gain (without that family L2 Δ −0.0024).
- Triad (Δ / new34 / nolidar): nolidar ≥ 0 everywhere; new34 is the blocker for L2/L2×m and P for all. W4's re-gate could only flip this if its harness differs, e.g. if it put the point inside finish or used a different p_scene source.
- Production cost if ever enabled: L2 needs the gor_gate DINOv2-S head (in the archive, OFF) and the dense probe (PCA + logreg, ~1 MB). L1 needs only the probe.

Files: `RESULT_PREREG_COMP_GORLOC.json` (full: per-fold picks, bootstrap, floors, LOFO, fold Δ), harness `work/gorloc/w3_comp.py` (snapshot `w3_comp.py.snapshot`), log `work/gorloc/both.log`.
No LB, no default.toml change, no flag enabled.
