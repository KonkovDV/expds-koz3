# koz3-w4-large-gate (CPU honest-10 · large_objects vs B′)

Plan **(b)** без fam0. Kernel script **~17 KB** (без embed); sidecars в датасете
`elenakonkova/koz3-w4-large-gate-code` (~3.2 MB: vendor/ijson, w5_rerank, scripts, ae OOF slim).

Варианты: `baseline` / `area_tau` / `fixc_nested` (τ_large=0.20). **Не сдавать на LB.**

## Push order (coord / bot-1; этот бот не пушит в Kaggle)
```bash
orch/pool_kaggle.sh elenakonkova datasets create -p /workspace/kaggle/push/large-gate-ds
# or: datasets version -m 'large-gate code+OOF slim' -p /workspace/kaggle/push/large-gate-ds
orch/pool_kaggle.sh elenakonkova kernels push -p /workspace/kaggle/push/large-gate
```

## Dry-run
```bash
LARGE_GATE_CODE=/workspace/kaggle/push/large-gate-ds \
LARGE_GATE_OUT=/tmp/lg_out \
  python3 /workspace/kaggle/push/large-gate/koz3-w4-large-gate.py
# → DONE MISSING_INPUT (нет w5base на боксе)
```
