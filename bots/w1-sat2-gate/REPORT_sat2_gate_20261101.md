# REPORT sat2 REPLACE honest-10 + sat_thr_relax — 01.11.2026

## Contract
| | |
|---|---|
| **goal** | Full honest-10 for sat2 REPLACE (soft-PASS → P≥0.9); tune `sat_thr_relax` scale on top |
| **hypothesis** | Soft-PASS 0.320516 ΔB′+0.00472 P0.838 reaches bootstrap P≥0.9; scale may lift P/Δ |
| **criterion** | Δ≥+0.003; **P = p_gt0 ≥ 0.9** (bootstrap P(improvement), not precision); damaged not worse; d_new34/d_optical ≥ −0.001 |
| **budget** | Kaggle CPU elenakonkova ~25–60 min; box cannot run (no stack_arch / no KEY_1) |
| **branch** | `bots/w1-sat2-gate` (lease bot-W1-sat) |
| **expected ΔLB** | **+0.003…+0.006** if pass_full; else 0; **hold 0.41238 — no submit** |

## Status
Package ready for bot1 push. Full honest-10 **not run on box** (missing `elenakonkova/koz3-w2-gate` w5base + KEY_1).

## Package
- Path: `/workspace/kaggle/push/sat2-gate/`
- Slug: `elenakonkova/koz3-w1-sat2-gate`
- Dataset: `elenakonkova/koz3-w2-gate`
- Manifest sha256: `d6323ebbdbe487f324c45b1894e16555868564bbde0b18bee9f4adb1eed6544c`
- Script sha256: `a5768699d1d51ccff97eaad6b0480e00060ba21c963c0b09d2e90380caf58818`
- OOF sha256: `d28feee138250ae18959c036454a86ab0c01724e0930a1ff74fad7f0e98161e6` (2622 feats, folds0–4 no-TTA)
- Soft-PASS Drive zip sha `c1e54f76…` — swap OOF if score ≠ 0.320516

## Method (kernel)
1. REPLACE: drop `sp_*` kurgan markup from stack_arch; append OOF as `threshold_group=sp_det2`
2. prep.py once
3. gate_v6c boot=4000 → `replace_all`
4. sat_thr_relax scales `{1.0…0.5}` via keep-mask on same table (no leakage of LB; nested honest per scale)
5. Best by (pass_full, p_gt0, d)

## P interpretation
Criterion «P ≥ 0.9» = **`boot.p_gt0`**. Soft-PASS P0.838 was the same quantity under soft bar P≥0.8.

## Base
Prefer `bprime_counts.pkl` (B′=0.315800). Else gate vs A `base_counts.pkl` and report `delta_vs_bprime_abs`.

## Prior / deferred
- Soft-PASS (bot1/Drive): score 0.320516, ΔB′ +0.00472, P0.838 — accepted soft.
- 10–50 m FN shift analysis: **deferred** (not finished this turn; not blocking sat2 gate).
- Closed: WSR, ring/near_kept, peak-rerank, sp_large, 8 m cluster — not redone.
- v2+FR1h gate: owned by bot1 — not duplicated.

## Bot1 action
```bash
orch/lease.sh who branch:bots/w1-sat2-gate
orch/pool_kaggle.sh elenakonkova kernels push -p /workspace/kaggle/push/sat2-gate
# optional: copy bprime_counts.pkl into package data/ before push
```
After COMPLETE: pull → `out/sat2-gate/`; fill EXPERIMENTS numbers; release leases.
