# PREREG W4 independent re-gate of W1 J2 (RR1s-cf + M_rp on top of v3_on) — bot5-w4, 30.10.2026 16:50 MSK, before any J2 measurement
Claim under test (W1, bots/w1-aeor 4293458): J2 vs V3P Δ +0.01561, P 0.9845, α_k taken from base A.
Read for this: W1 docs PREREG_W1/STATUS (4293458), W1 box code w1/cf/{ws_rr,fast,common}.py (blend + nested picks); data w1/cf/w5d/work/nested_w5cf.npz (outer/inner nested OOF).
W1 joint_gate.py NOT read (lives under analysis/**). M_rp here = W3 definition (disk r=0.05 m, as P-W4-compgate), not W1's w0 rp pairs — independent.

## Exactly one variant: J2W4
- Base B = V3P (PREREG_w4_v3on; compgate engine: table A − etgr − sp groups on D-FINE regions + C19 etgr + sp_dfine circles).
- + M_rp: all rows of gorodishcha/arkhitektury/object_poly/fortifikatsii (stack rows of V3P, C19, D-FINE circles) → disks, as compgate V3P_MRP.
- + RR1s-cf: for outer fold k, confidence of every stack row with markup_type li = blend(k, α) exactly as ws_rr.blend
  (per class rank of li rows RB; rows with nested score ≥ 0 get α·rank-mapped nested score + (1−α)·RB; nested score = outer OOF on fold k, inner[k] elsewhere). Other rows keep conf.
- α_k ∈ {0.3, 0.6, 1.0} re-picked ON v3 (J2W4 curves) per outer fold k: inner honest score over folds j ≠ k (train = all − j − k, joint group resweep), first max in list order. No in-sample choice, W1's α picks are not used (only reported).
- Thresholds: joint resweep on the 9 train folds with blend(k, α_k); only fold-k regions counted (D.honest, unseen=drop).
- Engine: d3_gate TableEngine/D.honest (official compute_metrics matching), family_folds 10 (asserted identical to ws_rr folds), paired family bootstrap 2000 seed 0.
## Sanity (must hold before verdict)
- K: split_arch key == table.pkl row index for all RPC rows (cls, conf, region equal).
- N: same per-fold pipeline with conf unchanged reproduces compgate V3P_MRP 0.3365449917 (|d| ≤ 1e-9).
## Gate J2W4 vs V3P
PASS iff Δall ≥ +0.003 AND P ≥ 0.9857 AND per-class floors (kurgany/gorodishcha ≥ −0.002, others ≥ −0.005) AND new34 Δ ≥ 0 AND nolidar Δ ≥ 0.
Secondary (report only): J2W4 vs V3P+M_rp; α picks vs W1 picks. Output: orch/bot-W4/runs/independent_regate_W1J2.json.
