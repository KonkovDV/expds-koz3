# koz3-bot7-geo-gate

Real honest-10 scoreboard for geo levers vs B′=0.315800 (Ae-ON flags-off).

**Push (bot1 only):**
```
orch/pool_kaggle.sh dmitrijkv kernels push -p /workspace/kaggle/push/geo-gate
```

Rebuild embed after sidecar edits: `python build_embed.py`

Env: `GEO_GATE_SKIP_S3`, `GEO_GATE_DRYRUN`, `GEO_GATE_REINFER`, `GEO_GATE_BOOT`, `GEO_GATE_W5BASE`, `BPRIME`.
