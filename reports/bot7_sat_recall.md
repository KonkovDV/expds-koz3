# bot7 sat-recall — real honest-10 (was stub)

## Audit
Prior package `dmitrijkv/koz3-bot7-sat-recall` (pushed ~20:01) used `prior_only` stubs → **EMPTY null scores**.
Fixed: prep→gate_v6c on w5base stack with post levers.

## B′
0.315800 Ae-ON flags-off honest-10 (same as geo-gate).

## Levers (all post-on-stack)
- L1 sat_band_verify (scale=0.90, verify_floor=0.40)
- L2 sat_agree_accept (radius_m=16 ≠ CLOSED 8m, scale=0.85)
- L3 sat_cross_nms (nms_m=4.0)
- ALL = L1+L2+L3

Note: L1/L2 post-hoc on already-thresholded stack may be near-noop without mid-band OOF; still emit real scores (honest Δ).

## Package
`/workspace/kaggle/push/sat-recall/` — push via `orch/pool_kaggle.sh dmitrijkv kernels push -p /workspace/kaggle/push/sat-recall`
