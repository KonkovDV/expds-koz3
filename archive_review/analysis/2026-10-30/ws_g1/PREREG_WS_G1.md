# PREREG WS-G1 — DINOv2 kNN verifier for gor/fort (bot C, 30.10.2026 13:19 MSK, BEFORE measurement)

Data: out/w5-dino-emb (dinov2_vits14 CPU, n=222929, ok=212101; cid = row in w5data/work/table.pkl). Embedded gor/fort rows: 14526
of 67192 (export mask; rows without embedding keep conf unchanged). Base A = honest-10 0.3146153226828423.
Label: candidate is a hit-able pred = has ≥1 official-match pair with a GT of its own class (table pairs) — "hit".
Score (honest, no leakage): for outer fold k, reference set = embedded gor/fort candidates of the SAME class on the other 9 folds
(family folds = gate folds). s = fraction of positives (hit) among k=5 nearest cosine neighbours (L2-normalised emb).
Multiplier: m = (0.2 + 1.6·s)^λ (s=0.5 -> 1.0, s=0 -> 0.2^λ, s=1 -> 1.8^λ); conf' = conf·m, applied only to gor/fort
rows with embedding. Grid λ ∈ {0.5, 1.0} (+ ref 2.0); primary = λ=1.0 fixed (no nesting; refs informational).
Thresholds re-swept per (class, group) on train folds (honest_counts), paired family bootstrap 2000 vs A, #12 public/closed reported.
PASS: Δ≥+0.003, P≥0.9, d_new34≥−0.001, d_optical≥−0.001, per-class floors (gor −0.002, others −0.005).
Variant G1rp: same on top of M_rp pairs (A+M_rp as base context; compared vs A AND vs M_rp) — only if M_rp re-PASSes here.
M_rp interaction: M_rp changes only which GT a gor/fort pred hits (rp disk), not conf; G1 changes conf only -> expected additive.
Prior expectation: +0.002…+0.008 (WEAKSPOTS_PLAN). Coverage risk: only ~22% of gor/fort rows embedded -> effect bounded.
Code: orch/bot-W5/scripts/ws_g1.py (measure-only; no archive code).

## Result 13:22 MSK: G1 (λ 0.5/1/2) FAIL Δ −0.020…−0.022, P 0.0 (runs/v3a_G1.json). Cause (diagnosis): positives are rare (gor 144/4272,
fort 285/10254 ≈ 3 %) so s≈0 for most rows -> m≈0.2 shrinks embedded rows below non-embedded rows of the same group.
## Addendum 13:23 MSK (before measuring G1n): base-rate-normalised multiplier
G1n: m = clip(s / b_k, 0.5, 2)^λ, b_k = positive rate of the train reference set of that class in outer fold k (ε: s=0 -> 0.5).
λ ∈ {0.5, 1.0}, nested pick (inner 9-fold honest), primary = `G1n_nested`. Same gate. If FAIL -> WS-G1 closed on ViT-S emb;
re-try only with ViT-B emb (out/w5-dino-vitb) as new data.
