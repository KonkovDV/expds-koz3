# koz3-tta-gate — TTA / multiscale honest-10 gate (bot-8)

**Slug:** `dmitrijkv/koz3-tta-gate`  
**Title:** `koz3-tta-gate`  
**Package push dir:** `/workspace/kaggle/push/tta-gate/`  
**Do not LB-submit.** Bot 1 pushes; run only after Ae gate result exists.

## CRITICAL: unique slug (do not collide with bot3)

This kernel id is **`dmitrijkv/koz3-tta-gate`**.  
**Never** use `dmitrijkv/koz3-ae-dfine-gate` — that slug belongs to **bot 3's Ae gate** kernel; pushing under it would overwrite their gate.

## Ae D-FINE weights — what to attach after Ae gate

Weights must come from the **accepted Ae gate output**, not a random FT dump.

| When | Attach | Version pin |
|---|---|---|
| **Until Ae gate finishes** (current metadata) | dataset `dmitrijkv/koz3-w1-ae-ft-f4pack` | f4pack sha prefix **`249b86fb`** (`dfine_ae/` + `thresholds_oof.json`); same pack ae2or / bot3 eval use |
| **After bot3 Ae gate PASS** | add kernel_source **`dmitrijkv/koz3-ae-dfine-gate`** (read-only attach of bot3 output) | Prefer gate-emitted weights/thresholds over the f4pack dataset if the gate promotes a newer pack; keep `249b86fb` as fallback |
| Always | `dmitrijkv/koz3-s3-cred` | S3 creds |
| Code/helpers | `dmitrijkv/koz3-dfine-ae` | Ae D-FINE kernel lineage (same as `koz3-dfine-ae-ft`) |

Bot 1 also attaches: mock/honest-10 data + repo snapshot (`bots/bot8-tta-gate` tip) when launching for real.

## Variants (acceptance vs B′ = 0.315800)

| id | overlays |
|---|---|
| `bprime` | all TTA/MS OFF (B′ flags-off baseline) |
| `tta_ae` | D-FINE Ae `rot90_tta=true` (+ flip as bot3 default) |
| `tta_ms_sat` | `[orsp_tta]` U-Net sat: `unet_tta=4`, `unet_scales=[0.75,1.0,1.25]` |
| `both` | Ae rot90 + scales + sat TTA/MS |

Gate: Δ≥0.003 vs B′, P≥0.9, no layer drop >0.01, damaged not worse. Wall target ≤20 min LB (see REPORT.md).

## Dry-run (box CPU)

```bash
TTA_GATE_DRY=1 TTA_GATE_OUT=/tmp/tta-gate-dry nice -n 10 python3 koz3-tta-gate.py
```

Resumable: `STATUS.txt` / `STATUS.json`, `variants/<id>/{done.json,score.json,wall.json}`, `SHA256SUMS`.
