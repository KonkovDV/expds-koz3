# pixgbm-cr — Kaggle CPU e2e geo F1 gate (bot-6)

Private CPU kernel. **Do not push from bot-6** — coord «1» creates datasets then pushes.

## Goal

REF (v2 `eb268d6` **output**) vs CAND (`submission_pixgbm_clean_e1g1_p4_lleaves.zip`)
on regions **200–217**: `|dF1| ≤ 0.01` on ALL + maggr/etgr/terfl via
`helpers/bot6_r3_gate.py`.

## Kernel

| Field | Value |
|---|---|
| id | `dmitrijkv/koz3-bot6-pixgbm-cr-gate` |
| owner | **dmitrijkv** |
| enable_internet | **true** (S3 + optional pip) |
| enable_gpu | **false** |
| code_file | `pixgbm_cr_gate.py` |
| default `GATE_MODE` | `auto` (REF output + CAND zip) |

## Geo data (S3)

Raw MagGr/EtGr/TerFl rasters + GT labels for 200–217 are pulled inside the kernel
from bucket `train-expds-3` (`https://s3.ru-7.storage.selcloud.ru`, region `ru-7`).
Cached under `/kaggle/working/train-expds-3` with skip-if-exists + size check.

Credentials: Kaggle Secrets `KOZ3_S3_ACCESS_KEY_ID` / `KOZ3_S3_SECRET_ACCESS_KEY`,
fallback dataset `dmitrijkv/koz3-s3-cred` (`cred.json`). **Never log keys.**

S3 API is an inline block in `rebuild_w4_geo_tree.py` marked
`BEGIN/END koz3_s3` — swap later with `bots/bot7-geo-gate` `kaggle/common/koz3_s3.py`.

`koz3-w4-geo-s000/s001` are **not** wired (do not exist).

## dataset_sources

```
dmitrijkv/koz3-s3-cred
dmitrijkv/koz3-pixgbm-cr-ref      # create: v2 result.geojson ONLY
dmitrijkv/koz3-pixgbm-cr-cand     # create: clean zip already staged
dmitrijkv/koz3-pixgbm-cr-wheels   # create: zstandard wheels
```

## Datasets «1» must create

1. `dmitrijkv/koz3-pixgbm-cr-wheels` ← `ds/pixgbm-cr-wheels/`
2. `dmitrijkv/koz3-pixgbm-cr-cand` ← `ds/pixgbm-cr-cand/` (zip present)
3. `dmitrijkv/koz3-pixgbm-cr-ref` ← `ds/pixgbm-cr-ref/` **after** dropping v2 `result.geojson` for 200–217
4. Confirm `dmitrijkv/koz3-s3-cred` exists and is attachable (or Secrets set on the kernel)

## REF contents

**Empty of v2 output on this box.** No `result.geojson` covering 200–217 was found
(bot2_v2 docker mock = `101_*`). Staging folder holds README + drop marker only.
EN `geo_pixgbm.py` / models were removed and must not be re-added.

## CAND

`submission_pixgbm_clean_e1g1_p4_lleaves.zip` sha256
`ed51cae4a2083830588f272613d4b97e108af439fa5c3f555fa135a56c902590` — verified.

## Offline deps

Prefer `koz3-pixgbm-cr-wheels` (`zstandard`). Network `pip install boto3` allowed
if wheels miss it. Inference stack expected on Kaggle CPU image.

## Local check

```bash
python pixgbm_cr_gate.py --syntax
```

## Push (coord «1» only)

```bash
kaggle datasets create -p ds/pixgbm-cr-wheels --dir-mode zip
kaggle datasets create -p ds/pixgbm-cr-cand   --dir-mode zip
# after REF geojson dropped:
kaggle datasets create -p ds/pixgbm-cr-ref    --dir-mode zip
kaggle kernels push -p .
```
