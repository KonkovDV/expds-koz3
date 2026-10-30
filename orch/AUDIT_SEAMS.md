# AUDIT_SEAMS (W0, 30.10, short version: budget cut)
Data: stack_arch OOF + w5data/work/table.pkl (it IS on the box); gate = W5 honest-10, 103 families. Code on bots/w0-seams@5171c35.

| # | Seam | Current (file:line) | Δ official | Fix / cost | Status |
|---|------|---------------------|-----------|-------------|--------|
| 1 | Anchor of micro-disk for non-kurgans | post.py:268 micro_circles (centroid; only kurgans) | **+0.00437** honest, P 0.9885, new34 +0.0022, opt +0.0085 | anchor="rp", r 0.05 m for gor/arch/obj/fort (pred → disk at rp(P): weakly dominant under the rule cov≥0.5 & rp∈G) | **code+test, gated PASS**; config commented, enable in the package after re-tuning thresholds |
| 2 | Fortifications are not micro (thin walls, cov<0.5) | default.toml:416 | +0.00205 P 0.982 (part of #1) | same | in #1 |
| 3 | Hillforts at rp vs polylabel/centroid | — | rp +0.0025 P 0.87 (new34 −0.0012); polylabel worse on new34 (−0.009) | do not use polylabel/centroid as the anchor | measured |
| 4 | Kurgans: centroid vs rp | post.py:296 | −0.0003 (noise) | leave as is | measured |
| 5 | Thresholds tuned before the shape change | watchdog.py:98 → 121 | ≥0 (dominance), up to +Δ#1 fully only after a re-sweep | re-sweep thresholds on OOF after micro(rp) | queue W5 |
| 6 | DEM without class 2 = mean of ALL points (DSM: trees/buildings) | lidar.py:258-262, 307-309 | n/a (count of such LAS unknown) | min-per-cell + PMF/SMRF ground filter when there is no class 2 | queue W3 |
| 7 | Confidence ties: kurgans 96–98% duplicated values | stack (upstream quantisation) | small; the greedy matcher and the threshold cannot split ties | do not round the scores, add a tie-breaker (e.g. +1e-6·rank of the second model) | queue W4 |
| 8 | cap_features: quota by class weight, not by marginal F1 | post.py:208 | 0 while the 20k cap does not bind | allocate by the marginal TP-rate of the threshold | low priority |
| 9 | >3000 vertices: writer drops, post simplifies | writer.py:139, post.py:119 | 0 (in the stack max nv<3000) | — | OK |
| 10 | Geodesy/pixel: zone by centroid, half-pixel, resampling | metric _to_utm L452; raster.py:338 (+0.5); tiles.py:387 (average on downsampling) | 0 (correct; cov is invariant to conformal scale) | — | OK |

Also checked: _ground_scale (post.py:262) = cos φ via the Gudermannian, correct; writer rounds to 0.01 units of 3857 ≈ 6 mm, harmless.
Artifacts: analysis/2026-10-30/w0_seams/{m1_anchor,g1_disk_gate,g2_decomp}.json on bots/w0-seams.
