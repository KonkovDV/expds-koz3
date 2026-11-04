# koz3-pixgbm-cr-wheels

Offline wheels for pixgbm-cr with `enable_internet=false`.

## Vendored

| Wheel | Why |
|---|---|
| `zstandard-0.25.0` cp310/cp311/cp312 manylinux2014_x86_64 | decode `small_files.tar.zst` from xfer-s000 when `zstd` CLI is missing |

## Not vendored (present on Kaggle CPU image)

`numpy`, `scipy`, `lightgbm`, `rasterio`, `shapely` — used by REF/CAND inference,
not by the gate script itself (stdlib only).

## Create

```bash
cd /workspace/kaggle/push/pixgbm-cr/ds/pixgbm-cr-wheels
kaggle datasets create -p . --dir-mode zip
```
