# koz3-w4-li-post-gate

CPU-only, internet-off honest leave-one-fold gate for **Li post levers**
(`[li_post]` class_tau + self_nms). Does **not** retrain. Does not touch closed
levers (r50c5 / WSR / ring / peak-rerank / sp_large / …). Not bot-7 geophysics.

## Inputs
- `elenakonkova/koz3-w2-oof` — ConvNeXt-Li OOF (`s0/` per-fold `oof_cnx_c3-sc_f*.geojson` + matching `gt_li_cnx_c3-sc_f*.geojson`).

## What it computes
For each held-out fold (thresholds fit on the other folds only):
- **OFF** — base τ=0.15 (production fallback when `thresholds_by_markup.li_cnx` absent)
- **thr_nested** — per-class absolute floors (grid)
- **nms8 / nms12** — same-class self-NMS radius 8 / 12 m
- **thr_nms8 / thr_nms12** — floors then NMS

Matching = official soft (`cov≥0.5` + representative point in GT).
`P = tp/(tp+0.5*fp)`. Outputs overall + per-class + per-size P/R/F1 and Δ vs OFF
→ `/kaggle/working/li_post_honest.json`.

## Expected runtime
- Full s0 5 folds on Kaggle CPU: **~5–15 min** (geometry soft-match).
- Box dry-run on light s0 copy: **~20–40 s**.

## Outputs
- `li_post_honest.json` — variants, thr_maps, by_cls, by_size
- `run.log` — STATUS lines

## Push (coord takes CPU lease)
```bash
orch/pool_kaggle.sh elenakonkova kernels push -p /workspace/kaggle/push/li-post-gate
```
Do **not** push without coord. Packaging does not require a lease.

## Dry-run (box)
```bash
LI_POST_OOF_ROOT=/workspace/kaggle/datasets/koz3-w2-oof-light \
LI_POST_OUT=/workspace/kaggle/push/li-post-gate/dryrun \
/workspace/koz3-venv/bin/python /workspace/kaggle/push/li-post-gate/koz3-w4-li-post-gate.py \
  2>&1 | tee /workspace/kaggle/push/li-post-gate/dryrun/dryrun.log
```

## Code flags (repo)
Branch `bots/bot4-li-post`: `[li_post]` in `configs/default.toml` (all OFF by default),
`expds/li_post.py`, hooked in `expds/watchdog.py` finish/safe_finish after apply_thresholds.
