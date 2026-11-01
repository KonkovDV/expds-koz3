# pixgbm-cr — Kaggle CPU e2e geo F1 gate (bot-6)

Private CPU kernel. **Do not push from bot-6** — coord «1» creates datasets then pushes.

## Goal

REF (v2 `eb268d6`) vs CAND (`impl=clean` / clean-pack) on regions **200–217**:
`|dF1| ≤ 0.01` on ALL + maggr/etgr/terfl via `helpers/bot6_r3_gate.py`.
INVALID on empty steps / no full / missing module / phase0 / docker≠0 /
geojson mtime before start / byte-identical results while model SHAs differ.

## Kernel

| Field | Value |
|---|---|
| id | `dmitrijkv/koz3-bot6-pixgbm-cr-gate` |
| enable_internet | **false** |
| enable_gpu | **false** |
| code_file | `pixgbm_cr_gate.py` |

Owner is **dmitrijkv** (owns `koz3-xfer-s000`, `koz3-gt-train-v2`; PREREG w4-geo slugs).
This box’s API token sees xfer/gt but gets **403** on `koz3-w4-geo-s000/s001` —
«1» must confirm those datasets exist and are readable by dmitrijkv (recreate or
grant access) before the gate can rebuild MagGr/EtGr/TerFl.

## dataset_sources (wired)

```
dmitrijkv/koz3-w4-geo-s000
dmitrijkv/koz3-w4-geo-s001
dmitrijkv/koz3-xfer-s000
dmitrijkv/koz3-gt-train-v2
dmitrijkv/koz3-pixgbm-cr-ref      # create from ds/pixgbm-cr-ref/
dmitrijkv/koz3-pixgbm-cr-cand     # create from ds/pixgbm-cr-cand/ after dropping CAND
dmitrijkv/koz3-pixgbm-cr-wheels   # create from ds/pixgbm-cr-wheels/
```

### Existing datasets searched

| Slug | Status |
|---|---|
| `elenakonkova/koz3-e2e-baseline-v2` | Documented REF zip host (e2e-eval). **403** from dmitrijkv token → not wired; tip-mode REF copied into `ds/pixgbm-cr-ref/` instead. |
| clean-pack / `models_pix_clean` on Kaggle | **Not found** → staging folder `ds/pixgbm-cr-cand/` (empty until «1» drops Drive zip). |
| `dmitrijkv/koz3-w4-geo-s000/s001` | Referenced by PREREG; API **403** / not in owned list — «1» action. |

## Push steps for «1» (exact order)

### 0. Fill CAND payload

```bash
# From Drive / PC orch sticky:
#   out/geo-clean-pack/submission_pixgbm_clean_e1g1_p4_lleaves.zip
cp /path/to/submission_pixgbm_clean_e1g1_p4_lleaves.zip \
   /workspace/kaggle/push/pixgbm-cr/ds/pixgbm-cr-cand/
# optional tip weights instead/in addition:
# cp models/pix_clean/* /workspace/kaggle/push/pixgbm-cr/ds/pixgbm-cr-cand/models_pix_clean/
(
  cd /workspace/kaggle/push/pixgbm-cr/ds/pixgbm-cr-cand
  find . -type f ! -name SHA256SUMS ! -name dataset-metadata.json ! -name README.md \
        ! -name 'DROP_*' ! -name 'README.txt' -print0 \
    | sort -z | xargs -0 sha256sum > SHA256SUMS
)
```

Optional zip-mode REF: also drop `submission_bot7cand_v2_eb268d6.zip` into
`ds/pixgbm-cr-ref/` (from `elenakonkova/koz3-e2e-baseline-v2`) and refresh SHA256SUMS.

### 1. Create datasets (first time)

```bash
export KAGGLE_CONFIG_DIR=...   # dmitrijkv credentials
cd /workspace/kaggle/push/pixgbm-cr

kaggle datasets create -p ds/pixgbm-cr-wheels --dir-mode zip
kaggle datasets create -p ds/pixgbm-cr-ref    --dir-mode zip
kaggle datasets create -p ds/pixgbm-cr-cand   --dir-mode zip
```

Later updates: `kaggle datasets version -p ds/... -m 'msg' --dir-mode zip`.

### 2. Confirm w4-geo readable

```bash
kaggle datasets view dmitrijkv/koz3-w4-geo-s000
kaggle datasets view dmitrijkv/koz3-w4-geo-s001
```

If 403/404: recreate or re-share under dmitrijkv, then re-check.

### 3. Push kernel (after datasets exist)

```bash
cd /workspace/kaggle/push/pixgbm-cr
# kernel-metadata.json already lists all dataset_sources; enable_internet=false
kaggle kernels push -p .
```

### 4. Optional env

| Var | Default | Meaning |
|---|---|---|
| `GATE_MODE` | `tip` | `tip` (REF/CAND tip packs) \| `zip` \| `score` |
| `PIXGBM_DF1_TOL` | `0.01` | equivalence tolerance |
| `PIXGBM_CR_DRY_RUN` | — | `1` → syntax/import check only |

Local check: `python pixgbm_cr_gate.py --syntax`

## Offline deps

Gate script: **stdlib only**.

Vendored (dataset `koz3-pixgbm-cr-wheels`):

- `zstandard-0.25.0` wheels cp310/cp311/cp312 manylinux2014_x86_64  
  Installed only if needed via `pip install --no-index --find-links …` (no network).

Expected already on Kaggle CPU image (not vendored):

- `numpy`, `scipy`, `lightgbm`, `rasterio`, `shapely`

`geo_pixgbm.py` lives **only** under `ds/pixgbm-cr-ref/` (runtime black-box).
It is gitignored as `expds/geo_pixgbm.py` and must **never** be committed from this package.

## Package layout

```
pixgbm-cr/
  kernel-metadata.json
  pixgbm_cr_gate.py
  rebuild_w4_geo_tree.py
  README.md
  helpers/bot6_r3_*.py
  scripts/rebuild_train_tree.py
  notes/SPEC.json
  ds/
    pixgbm-cr-ref/     # ~4.2M tip REF (+ optional zip)
    pixgbm-cr-cand/    # staging until clean-pack dropped
    pixgbm-cr-wheels/  # ~16M zstandard wheels
```

## Conflict note

`9b411c4` / PR #20 = EN-lineage P-retrain tooling, **not** clean-room.
Preferred CAND = `impl=clean` + `models/pix_clean` / clean-pack zip.
