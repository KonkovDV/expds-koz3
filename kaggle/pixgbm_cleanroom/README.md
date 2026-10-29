# Kaggle kernel: PIXGBM clean-room retrain (TPU-VM as CPU host)

Bot-6 / BACKLOG TPU-2. LightGBM only (no torch_xla). TPU accelerator selected so the
session gets a large CPU/RAM host; training uses configurable `PIXGBM_NUM_THREADS`.

## Enable flag (default OFF)

The repo script `scripts/pixgbm_retrain_cleanroom.py` refuses to train unless
`--enable-retrain` is passed. This kernel always passes that flag when run as a
job; do not merge enabling into submission configs.

## Data (OPEN QUESTIONS for bot 1)

| Item | Status |
|---|---|
| Full КОЗ №3 train root with geo regions 200–217 | **OPEN**: no dedicated public/private Kaggle dataset slug confirmed for full `train-expds-3` rasters. Other kernels use S3 via `dmitrijkv/koz3-s3-cred`. |
| Expected local layout after sync | `<ROOT>/<region>*/` with maggr/etgr/terfl rasters + ObjectPoly/FindsPoints annotations (same as `F:\2\train-expds-3`). |
| Proposed dataset slug if packaging a subset | `dmitrijkv/koz3-train-geo-200-217` (does not exist yet — create only if DATA_PROTOCOL allows Kaggle copy). |
| `expds/geo_pixgbm.py` | gitignored on main; kernel looks for `/kaggle/input/*/geo_pixgbm.py` or `REPO/expds/geo_pixgbm.py`. Attach build-time copy sha256 `35e4331b…` (see `docs/PIXGBM_DECISION.md`). |

Env overrides (bot 1 fills after TPU probe):

- `PIXGBM_NUM_THREADS` (default 8; use 1 for max bit-exact chance)
- `PIXGBM_ROUNDS` (default 300)
- `PIXGBM_DATA_ROOT` (default: first existing among `/kaggle/input/*/train-expds-3`, `/kaggle/working/train-expds-3`)
- `PIXGBM_METRIC_TOL` (default `1e-4`)

## Outputs (`/kaggle/working/pixgbm_retrain/`)

- `STATUS.txt` — RUNNING/DONE/FAILED + ISO time
- `retrain_log.json` — verdict vs MODELS_NOT_IN_GIT.md
- `models/all/poly.txt`, `models/all/finds.txt` + `*.sha256`
- `checkpoints/` — after extract, after each held-out family
- `logs/train.log`

## Pass criterion

`retrain_log.json` status `ok` (sha256 match) **or** `metric_fallback.status == ok` within tolerance.
