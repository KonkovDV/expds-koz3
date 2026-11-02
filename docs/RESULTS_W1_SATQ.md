# RESULTS W1-SATQ (frozen; PREREG 0614536 + Amendment 1 d425325), 02.11.2026, bot-W1-satq

Production replay of v3_on b447649 (configs/v3_on.toml), honest-10 nested over t_taus folds, paired site-family bootstrap
(2000, seed 0). Base PH 0.345439 (replayed 0.3454388 ✓), gate-side base 0.3337867 ✓, freeze rule asserted in every fold
(non-sp_dfine τ = base; unchanged sp_dfine classes = base; identity refit mismatches 0). Family m = 6, Bonferroni P ≥ 0.98333;
Q1* P ≥ 0.99. Scripts: tools/satq/satqf_{prep,replay,score}.py, run_satqf.sh (4d5420d). Outputs /workspace/koz3/w1/satq/frzq/.

| variant | Δall | P | thr | Δnew34 | Δnolidar | P4 | verdict |
|---|---|---|---|---|---|---|---|
| Q1a prior arkh | +0.004695 | 0.9735 | 0.99 | 0.0000 | +0.004684 | ok (arkh +0.047) | **FAIL** (P) |
| Q1b prior × τ | +0.004854 | 0.9745 | 0.99 | 0.0000 | +0.004848 | ok (arkh +0.049) | **FAIL** (P) |
| Q1c τ only | +0.004367 | 0.9835 | 0.99 | 0.0000 | +0.004478 | ok (arkh +0.044) | **FAIL** (P) |
| FP1 (0.5, 0.5) | **+0.008581** | **0.999** | 0.98333 | **+0.000896** | **+0.018988** | ok (kp +0.0115, kt +0.0020, arkh +0.0474, fort −0.00007) | **PASS** |
| FP2 nested | +0.005269 | 1.0 | 0.98333 | −0.000697 | +0.010671 | ok (kp +0.0096, kt −0.0012, arkh +0.026) | **FAIL** (new34) |
| Q2 ≥4 Mpix gate | +0.008715 | 0.9955 | 0.98333 | −0.001324 | +0.009561 | ok (kp +0.0020, kt −0.0013, gor +0.0168, arkh +0.0586) | **FAIL** (new34) |

Per-fold choices: Q1a (d0, s) = (0.25, 1.5) ×9, (0, 1.0) fold 2. Q1b (d0, s, τ) = (0.25, 1.5, 0.30) ×8, (0, 1.0, 0.30) folds 2, 6.
Q1c τ = 0.373 ×10. FP2 (d0, edge) = (0.25, 0.5) ×8, (0.25, 0.25) fold 0, (0.75, 0.75) fold 3. Q2 (x, t2) = (0.4, 0.85) ×8,
(0.5, 0.85) fold 0, (0.6, 0.85) fold 7 (t2 at the top of the grid in every fold).

Gate-side nested (frozen): Q1a +0.003118, Q1b +0.003203, Q1c +0.003710, FP1 +0.007715, FP2 +0.004413, Q2 +0.007638.
Change vs the pre-freeze disclosure: Q1c +0.000072 → +0.003710, because the tie rule is now as written in the PREREG
("larger τ"); the pre-freeze q1_prep.py preferred the smaller τ (code bug, disclosed). FP1/FP2 changed through the freeze.

Notes: FP1 = v3_on's own frame_prior parameters rebound to sp_dfine (fixing the no-op), with no selection. Q2 and FP2 lose on
new34 through kt (Q2: the post-hoc 4 Mpix cutoff from W4 Q0 does not rescue new34). Q2 production code
[post].sat_frame_gate + scene_mpix is in place and OFF by default (tests pass).
