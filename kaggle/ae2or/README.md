# koz3-ae2or-ft — D-FINE Ae → Or transfer fine-tune

**Owner:** bot3 (`bots/bot3-ae2or`). **Do not push without coord lease** (bot 1 pushes).

## Goal
Transfer fine-tune Ae FT weights onto orthophoto (`Or`) leave-region-out folds; emit `dfine_or/` + honest OOF `thresholds_or_oof.json` (Fβ=2) for `[dfine_aeor]` `markups=["or"]` (default OFF).

## Init weights
- Primary: Ae FT **f4pack** sha `249b86fb` → dataset `dmitrijkv/koz3-ae-ft-f4pack` (`dfine_ae/`).
- Alt: final `3e619878` via `DFINE_INIT=/kaggle/input/.../dfine_ae`.
- Env override: `DFINE_INIT`.

## Why dfine-or ≈ 0 (bot6) — mitigations here
| Failure | Mitigation |
|---|---|
| Non-arch soft TP = 0; provisional p25 τ FP flood | Honest OOF Fβ=2; τ=1.01 if OOF tp=0 |
| Ae `OFF_CLASSES` would kill arch | `OFF_CLASSES=set()` for Or |
| Tile 1024@0.5 from-scratch scale vs Ae 640 | Keep **640/160 @ 0.5 m** (Ae-matched) |
| Domain/empty chips risk | Ae init + Or RGB @ same GSD path |
| Nested F1 ≪ surf_arch | Transfer + recall-oriented τ; enable only after honest-10 Δ>0 |

## GSD / bands / classes
- Or native ~1–2 cm → prep ~0.25 m → Scene `WarpedVRT` **average → 0.5 m** (same as Ae FT).
- Bands: RGB 3ch (Or COG), same as Ae.
- Classes: shared 5 (`kurgany_tselye|povrezhdennye|gorodishcha|fortifikatsii|arkhitektury`).

## Folds (leave-region-out, no family leak)
- f0: 008_Герите, 014_Лешки, 093_Старый_Барах
- f1: 033_Елизаветинское, 056_Фалхан
- f2: 185_Изборск

## Budget (lean T4×2)
| Setting | Default |
|---|---|
| iters/fold | 1200 |
| final_iters | 800 |
| lr | 5e-5 |
| deadline_h | 4.0 |
| ckpt_every | 400 |
| **est. GPU hours** | **~3.5–4.5 h** hard deadline 4.0 h |

## Outputs
- `dfine_or/` + `models/dfine_or/` (HF safetensors)
- `thresholds_or_oof.json` (+ detail) under working + `models/dfine_aeor/`
- `oof_dfine_f{k}.geojson`, `dfine_report.json`, `STATUS.txt|json`, `PACK_HINT.toml`

## Dry-run (box CPU)
```bash
DFINE_DRYRUN=1 DFINE_MODE=smoke DFINE_OUT=/tmp/ae2or_dry \
  python koz3-ae2or-ft.py --allow-cpu
```
Exercises Or folds, mid-ckpt resume (`find_fold_ckpt`), OOF Fβ=2 calib, packaging.

## Pipeline slot (default OFF)
See `PACK_HINT.toml`. When enabling later with A+B+C, raise `budget_s` (infer was 3:48/30) but keep total ≤20 min.

## RULES
No Yandex/Google/Bing/Esri; no КОЗ №2. Branch `bots/bot3-*` only; never main.
