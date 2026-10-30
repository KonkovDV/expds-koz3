# Gate prep inventory — v2/geo5 compare (C, 30.10 14:30 MSK)

Interim per A: locate/pull geo5 + bot7cand_v2 geo harness. **No aaaa/cpu-2. v3a_min frozen. No LB.**

## Target gate (from LB_DROP / official)
| | v2 bot7cand eb268d6 | v3a_min (failed LB) |
|---|---|---|
| geo P/R | **.769 / .656** | .716 / .311 |
| final | 0.41238 | 0.37838 |

Gate rule for next Elena package: geo P/R ≥ v2 on comparable eval (**geo5 / modality**), baseline = bot7cand_v2 / geo5, **not** honest-10 A.

## FOUND on box
| Asset | Path |
|---|---|
| geo5 folds | `scripts/pix_clean_eval.py` FOLDS F1–F5 (200–217) |
| Proxy gate harness (vs A score2) | `wt/w4-pix/scripts/pix_clean_eval.py` + `pix_clean_oof.py compose` |
| Prior geo5/ORG JSON results | `wt/w4/orch/bot-W4/w4geo/eval/R17*.json`, `FINAL19_gate_N0.json` |
| Eval runner scripts | `wt/w4/orch/bot-W4/w4geo/run_orget_eval.sh` (+2) |
| Official modality metric | `wt/w4-pix/materials/compute_metrics.py` (markup_type P/R) |
| GT | `w5data/gt_train_v2.geojson` |
| Cleanroom SPEC | `orch/spec/PIXGBM_SPEC_CLEANROOM.md` |
| LB logs | `wt/w5/docs/background/LB_SCORING_v2_eb268d6_0.41238.log`; `/tmp/lb_zip/scoring.log` (v3a_min) |
| S3 train creds | `local_ctx/s3_train.env` |
| Geo key list (regen) | `s3_keys_geo_200_217.tsv` — **320 files, 1.43 GiB** |
| Fetch | `w4geo/fetch_geo.py` → `w4geo/train/` (started 14:30) |

## MISSING (blockers)
| Asset | Notes |
|---|---|
| `w4geo/train` rasters | pulling from S3 now |
| `w4geo/oof` + composed `A17*.geojson` / `C17*.geojson` | wiped with box; rebuild after train pull via `pix_clean_oof.py compose` |
| `w4clean` / `w4venv` | use `wt/w4-pix` + `/workspace/koz3/venv` |
| `submission_bot7cand_v2_eb268d6.zip` | **only on Elena PC** (parity note); need mirror for full-stack modality compare via compute_metrics |
| v2 EN PIXGBM preds on geo5 | not on box; cannot byte-compare EN without zip or frozen OOF |
| Modality P/R harness vs v2 | **gap**: `pix_clean_eval` gates Δ score2 vs A, not geo P/R vs v2 LB. Need thin wrapper: run `compute_metrics` on geo-only preds (or full stack) and assert P/R ≥ .769/.656 |
| `orch/PIXGBM_RECOVER_20261030.md` | not landed yet |

## Proposed C GO (when brief lands)
1. Finish S3 geo pull; rebuild OOF compose for C-et+TerFl maps=`["etgr","terfl"]` + thr terfl_pix=0.58017.
2. Add `scripts/geo_pr_vs_v2.py` (or extend pix_clean_eval) → modality geophys P/R using compute_metrics; target ≥ v2.
3. Gate every recover candidate vs that bar before any Elena zip.
4. Still need Elena/A to drop bot7cand_v2 zip (or geo-modality reference preds) if full-stack parity required.

