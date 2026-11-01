# koz3-ae-dfine-gate — TTA / multiscale honest-10 gate (bot-8)

**Slug:** `dmitrijkv/koz3-ae-dfine-gate`  
**Package push dir:** `/workspace/kaggle/push/tta-gate/`  
**Do not LB-submit.** Bot 1 pushes; run only after Ae gate result exists.

## Sources (mirrored from `koz3-dfine-ae-ft` / Ae lineage)

Slug `koz3-ae-dfine-gate` was **not** present as a prior package on the box (2026-11-01);
metadata mirrors `dmitrijkv/koz3-dfine-ae-ft`:

- `dataset_sources`: `dmitrijkv/koz3-s3-cred`
- `kernel_sources`: `dmitrijkv/koz3-dfine-ae`
- GPU T4, internet ON, same private BYOD docker digest

Attach additionally (bot 1): mock/honest-10 data + f4pack Ae weights + repo snapshot
(`bots/bot8-tta-gate` @ tip) when launching for real.

## Variants (acceptance vs B′ = 0.315800)

| id | overlays |
|---|---|
| `bprime` | all TTA/MS OFF (B′ flags-off baseline) |
| `tta_ae` | D-FINE Ae `rot90_tta=true` (+ flip as bot3 default) |
| `tta_ms_sat` | `[orsp_tta]` U-Net sat: `unet_tta=4`, `unet_scales=[0.75,1.0,1.25]` |
| `both` | Ae rot90 + sat TTA/MS |

Gate: Δ≥0.003 vs B′, P≥0.9, no layer drop >0.01, damaged not worse. Wall target ≤20 min LB.

## Dry-run (box CPU)

```bash
TTA_GATE_DRY=1 TTA_GATE_OUT=/tmp/tta-gate-dry python koz3-tta-gate.py
```

Resumable: `STATUS.txt` / `STATUS.json`, `variants/<id>/{done.json,score.json,wall.json}`, `SHA256SUMS`.
