# Kaggle kernel: PIXGBM clean-room retrain (TPU-VM as CPU host)

Bot-6 / BACKLOG TPU-2. LightGBM only. TPU accelerator → large CPU/RAM host.

## Datasets (closed: no single train-expds-3 slug)

| Dataset | Role |
|---|---|
| `dmitrijkv/koz3-xfer-s000` | `small_files.tar.zst` + manifest (UTM, разметка, extra; labels for geo 200–217) |
| `dmitrijkv/koz3-xfer-s001` … `s009` | large `f####.tif` / `f####.laz`; **per-shard manifest** maps blob → original path |
| `dmitrijkv/koz3-gt-train-v2` | consolidated GT overlay (optional for PIXGBM; ObjectPoly/FindsPoints already in s000) |

Kernel rebuilds `/kaggle/working/train-expds-3` via `rebuild_train_tree.py` (geo 200–217 only), then calls `scripts/pixgbm_retrain_cleanroom.py --enable-retrain`.

Offline copy of s000 manifest: `/home/box/shared/koz3/kaggle/koz3-xfer-s000_manifest.json`.

### Manifests still needed offline (ask bot 1)

To know **which** of s001–s009 hold MagGr/EtGr/TerFl for regions 200–217 without attaching all:

- `koz3-xfer-s001` … `koz3-xfer-s009` **manifest.json** (each shard).

Until then metadata attaches **all** s000–s009 (as instructed).

## Env

| Var | Default | Meaning |
|---|---|---|
| `PIXGBM_NUM_THREADS` | `os.cpu_count()` | LightGBM threads |
| `PIXGBM_ROUNDS` | 300 | boosting rounds |
| `PIXGBM_METRIC_TOL` | 1e-4 | sha mismatch fallback |
| `PIXGBM_TREE` | `/kaggle/working/train-expds-3` | rebuild destination |
| `PIXGBM_REBUILD_ALL` | unset | if `1`, materialize all regions (not only 200–217) |
| `PIXGBM_OUT` | `/kaggle/working/pixgbm_retrain` | models + STATUS |

## Outputs

`STATUS.txt` (RUNNING/DONE/FAILED), `retrain_log.json`, `models/all/{poly,finds}.txt`+`.sha256`, `checkpoints/rebuild_report.json`, `logs/train.log`.

## Pass

sha256 == MODELS_NOT_IN_GIT.md **or** metric_fallback.status=ok.
