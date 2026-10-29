# koz3-dfine-ae-ft — D-FINE Ae fine-tune (bot-3 / GPU-3)

Lean **resume** of completed kernel `dmitrijkv/koz3-dfine-ae` (obj2coco, Ae, family k5).
Produces weights for `[dfine_aeor]` and recall-oriented OOF thresholds. Does **not** enable any
inference flag (packaging hint keeps `enabled = false`).

## Outputs (`/kaggle/working`)

| path | purpose |
|---|---|
| `dfine_f{0..4}/` | per-fold HF dirs (`config.json` + `model.safetensors`) |
| `dfine_ae/` | all-data final HF dir — drop-in for `weights.ae = ["models/dfine_ae"]` |
| `models/dfine_ae/` | copy of final for submission layout |
| `models/dfine_aeor/thresholds_oof.json` | `{"dfine_ae": {class: τ}, "ae_dfine": …}` |
| `thresholds_ae_oof.json` + `_detail.json` | same + F_β detail |
| `oof_dfine_ae.geojson` | OOF detections |
| `dfine_report.json`, `STATUS.txt` / `STATUS.json` | progress / DONE |
| `PACK_HINT.toml` | how to wire into `configs/default.toml` (still OFF) |
| `ckpt_f{k}/`, `ckpt_final/` | mid-training checkpoints (every `DFINE_CKPT_EVERY`) |

## Env (defaults = lean FT)

| var | default | note |
|---|---|---|
| `DFINE_MODE` | `full` | `smoke` for dryrun |
| `DFINE_ITERS` | `2000` | **additional** iters per fold after resume |
| `DFINE_FINAL_ITERS` | `1500` | all-data final for `dfine_ae/` |
| `DFINE_DEADLINE_H` | `3.0` | wall budget incl. infer + calib |
| `DFINE_CKPT_EVERY` | `500` | mid-ckpt interval |
| `DFINE_RESUME` | (empty) | optional root containing `dfine_f{k}` |
| `DFINE_CALIB_BETA` | `2.0` | F_β for threshold sweep (recall-oriented) |
| `DFINE_FOLDS` | `3,4,0,2,1` | same family folds as completed Ae |

Resume order per fold: `/kaggle/working/ckpt_f{k}` → `/kaggle/input/**/dfine_f{k}` (from kernel_sources).

## GPU estimate (T4×2)

- chips+setup ≈ 0.3–0.5 h
- 5 folds × 2000 × ~0.98 s / 2 GPU ≈ 1.4 h
- OOF infer ≈ 0.3 h
- final 1500 iters ≈ 0.4–0.5 h
- calib CPU-light ≈ minutes  
**Total ≈ 2.5–3.0 GPU-h** (fits remaining ~9.4 h to 31.10 03:00 МСК with other jobs).

## RULES

- No Yandex / Google / Bing / Esri imagery; no КОЗ №2 data.
- Base weights: `ustc-community/dfine-large-obj2coco-e25` @ pinned rev (Apache-2.0) → fine-tuned on KOZ3 Ae only.
- Inference `[dfine_aeor]` stays default-OFF until honest-10 Δ>0 on PC.

## Operator

```bash
# after push of this folder as kernel source:
KAGGLE_API_TOKEN=… kaggle kernels push -p kaggle/2026-10-29/dfine_ae_ft
# require COMPLETE on dmitrijkv/koz3-dfine-ae (kernel_sources)
```

Success: `STATUS=DONE`, `dfine_ae/model.safetensors` present, `thresholds_ae_oof.json` present, no fold errors in `dfine_report.json`.
