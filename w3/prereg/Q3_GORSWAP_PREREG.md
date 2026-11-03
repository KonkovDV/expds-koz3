# Q3 GOR-SWAP PREREG — W3, 02.11 MSK. Committed before measurement. Spec: orch/COMPETITOR_GAPS_2026-11-02.md §3 Q3 (d2fe1de). Clean-room.
Flag gor_swap, default off. Base: V3P (v3_on b447649, honest-10 0.333787). One family, one variant with a nested grid.
- p_scene: the DINOv2-S hillfort-gate OOF patch p (bot-5 analysis/gor_patch/scores_cand7.json, out-of-family). p_scene is the max over the scene's scored patches.
- Scene: a D-FINE region with exactly one SpOr tif in bot-5 meta.json. Size = bbox of vignette corners / gsd² (native Mpx; an approximation because tifs are not on the box). Multi-tif regions and regions without p are untouched.
- Rule (replace mode), for a small scene (Mpx < A): drop all sp_dfine gor rows in the region.
  - If p ≥ g, emit one gor with conf = p, threshold group gor_swap. Its location is the best sp_dfine gor circle with score ≥ s_min within ρ·half of the scene centre; if there is none, a circle of median sp_dfine gor radius at the centre.
- Grid: A ∈ {2, 4, 8}, ρ ∈ {0.2, 0.35}, s_min ∈ {0.02, 0.1}, g ∈ {0.05, 0.1, 0.2, 0.3}, plus off. Chosen per outer fold on train (train-swept thresholds); ties prefer off, then grid order.
- Gate (3 simultaneous W3 gates Q3/Q4/Q5, Bonferroni):
  - Δ ≥ 0.003, P ≥ 1 − 0.1/3;
  - new34 ≥ 0, nolidar ≥ 0;
  - class floors kp/kt/gor −0.002, others −0.005.
- Diagnostic only (not a criterion): Δ without the Kislovodsk family (LOFO cluster check).

## Amendment 1 (02.11, before any Q3 measurement; Q3 had not produced an outcome)
The sanity check failed: only 61% of sp_dfine rows fall inside the vignette-corner bbox, because the vignettes do not cover the whole tif and the tifs are not on the box.
The scene extent is now bbox(vignette corners ∪ centroids of all D-FINE OOF circles in the region). The centre is the middle of that bbox, and half = min(w, h)/2. Mpx = w·h/gsd²/1e6.
Assert: ≥ 99% of sp_dfine gor rows inside the extent. Nothing else changes.
