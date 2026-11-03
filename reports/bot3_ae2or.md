# bot3 — Ae→Or D-FINE transfer FT package (01.11.2026 ~19:15 MSK)

| Field | Value |
|---|---|
| Branch | `bots/bot3-ae2or` (lease alias `bots/w1-ae2or` → owner `bot-W1-aeor`) |
| Package | `/workspace/kaggle/push/ae2or/` |
| Kernel slug | `dmitrijkv/koz3-w1-ae2or-ft` (**not pushed**; bot 1) |
| Init | Ae FT f4pack **249b86fb** (`out/dfine-ae-ft-f4pack/dfine_ae`); alt final 3e619878 |
| Dry-run | CPU `DFINE_DRYRUN=1` **DONE** (folds/resume/OOF Fβ=2/pack) |
| Est. GPU | **~3.5–4.5 h** T4×2; `deadline_h=4.0`; 3×1200 + final 800 @ lr 5e-5 |
| Leases | `branch:bots/w1-ae2or`, `kernel:dmitrijkv/koz3-w1-ae2or-ft`, `dataset:dmitrijkv/koz3-w1-ae-ft-f4pack` → bot-W1-aeor |

## Why dfine-or gave ~0 (bot6) — package fixes

From `reports/bot6_ortho_multiclass.md` + OOF:

1. **Non-arch soft TP = 0** (gor/fort/kurgan); provisional p25 τ flooded FP → nested wF1 ~0.037; deploy arch-only @0.2 ≈0.049.
2. **Arch F1 ≪ surf_arch** — domain/threshold, not CRS (median dist high but soft_tp@2m>0).
3. **From-scratch Or @1024/288** scale ≠ Ae feature scale.
4. Applying Ae **OFF_CLASSES** (`arkhitektury`) would kill the only live class.

**Mitigations in ae2or:** Ae FT init; Or leave-region-out f0–f2 (same sites); tile **640/160 @ 0.5 m** (Or prep→0.25 then WarpedVRT average→0.5, Ae-matched); RGB 3ch; shared 5-class map; `OFF_CLASSES=∅` + post-calib τ=1.01 if OOF tp=0; honest Fβ=2 → `thresholds_or_oof.json`; resumable STATUS/mid-ckpt.

## Expected LB delta (explicit assumptions)

Baseline scoring_v2 **eb268d6**: **ортофото P=0.368421 R=0.140000** (layer soft contribution to ALL 0.41238).

Assumptions (conservative):

| Assumption | Value |
|---|---|
| Or GT share of score mass | small vs lidar/geo; Or R=0.14 is the gap |
| Recoverable FN | mostly **arkhitektury** misses; bot6 nested arch F1~0.11–0.15 vs surf_arch dominant |
| Transfer lifts Or soft R | **0.14 → ~0.18–0.22** if Ae prior helps arch localization (not proven) |
| Soft P held | ≥0.30 (else FP hurts merge) |
| Enable gate | only if honest-10 Δ≥0.003 and no layer drop >0.01 |

**Expected ΔLB if ON after gate:** **+0.002 … +0.008** (Or layer only). Not enough alone to clear 0.41238→0.42; stacks with lidar/sat2. **If non-arch stay OFF and arch τ~0.2–0.5**, risk of FP vs surf_arch → Δ can be **≤0 or negative** (bot6 e2e warning). Treat as **optional OFF lever**, not primary beat-EN path.

FN recoverable (order-of-magnitude): Or R=0.14 ⇒ ~86% FN among Or GT; if 20–30% of arch FN become soft-TP without proportional FP, R→~0.18–0.20. Explicit: **no claim without honest-10**.

## Dry-run result

```
фолды OK: 6 участков, 6 семей [f0 008/014/093, f1 033/056, f2 185]
resume OK: find_fold_ckpt(0)= …/ckpt_f0
DRYRUN DONE τ: non-arch 1.01, arkhitektury 0.7; STATUS=DONE; models/dfine_or/ present
```

Artifacts: `kaggle/ae2or/dryrun_*`, push dir copies.

## Queue entry (for bot1)

```json
{
  "id": "S-ae2or",
  "owner": "bot-W1-aeor",
  "accel": "gpu",
  "account": "dmitrijkv",
  "kernel_dir": "/workspace/kaggle/push/ae2or",
  "ttl_h": 5,
  "status": "hold",
  "after": ["ae-dfine-gate"],
  "pull_to": "out/ae2or",
  "note": "Ae→Or FT; need dataset koz3-w1-ae-ft-f4pack; DO NOT collide with koz3-ae-dfine-gate"
}
```

## RULES

No Yandex/Google/Bing/Esri; no КОЗ №2; never main; inference slot default OFF.
