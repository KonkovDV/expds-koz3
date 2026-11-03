# koz3-bot7-sat-recall

CPU honest-10 gate package for satellite recall levers (bot7). **Do not push** from bot7 — bot1 only.

## Slug
`dmitrijkv/koz3-bot7-sat-recall`

## Push (bot1)
```bash
orch/pool_kaggle.sh dmitrijkv kernels push -p /workspace/kaggle/push/sat-recall
```
Secrets: `KOZ3_S3_ACCESS_KEY_ID`, `KOZ3_S3_SECRET_ACCESS_KEY` (or dataset `dmitrijkv/koz3-s3-cred`).
Internet ON, GPU OFF. Runtime: S3 pull ~10–40 min + prior checkpoints; full re-infer when `SAT_RECALL_REINFER=1` (not wired in stub — attach tip + stack).

## Gate vs B′ 0.315800
- Δ ≥ 0.003
- P = p_gt0 ≥ 0.9
- kurgany_povrezhdennye not worse

## Levers (tip default OFF)
| Lever | Flag | Prior E[ΔLB] |
|---|---|---|
| L1 | `sat_band_verify` scale=0.90 verify_floor=0.40 | +0.005 |
| L2 | `sat_agree_accept` r=16m scale=0.85 | +0.003 |
| L3 | `sat_cross_nms` nms_m=4 | +0.0015 |
| ALL | L1+L2+L3 | +0.007 |

Smoke: `SAT_RECALL_SMOKE=1 python koz3-bot7-sat-recall.py`
