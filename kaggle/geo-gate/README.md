# koz3-bot7-geo-gate — CPU + S3 honest-10 geo levers

**Не пушить с bot7.** Push (bot 1):

```bash
orch/pool_kaggle.sh dmitrijkv kernels push -p /workspace/kaggle/push/geo-gate
```

**Secrets на ядро:** `KOZ3_S3_ACCESS_KEY_ID`, `KOZ3_S3_SECRET_ACCESS_KEY`  
(или датасет `dmitrijkv/koz3-s3-cred`).

| | |
|---|---|
| **slug** | `dmitrijkv/koz3-bot7-geo-gate` |
| **internet** | **true** (S3) |
| **GPU** | false (CPU) |
| **B′** | 0.315800; gate Δ≥0.003, P≥0.9, damaged not worse |
| **runtime** | smoke ≪1 мин; S3 pull ~10–40 мин (≈1.4 ГиБ); post L3 ~20–60 мин; full re-infer L1/L2 **2–6 ч** CPU |

## dataset_sources

| Dataset | Status |
|---|---|
| `dmitrijkv/koz3-gt-train-v2` | **verified** ready |
| `dmitrijkv/koz3-xfer-s000` | **verified** ready (UTM/small) |
| `dmitrijkv/koz3-s3-cred` | **unverified** (API 403 from this token; docs + Secrets primary) |
| ~~`koz3-w4-geo-s000/s001`~~ | **missing** → S3 |
| ~~`elenakonkova/koz3-w2-gate`~~ | **unverified/403** → attach stack manually or omit |

## S3 keys (bucket `train-expds-3`)

Prefixes `200_`…`217_`; keep keys matching MagGr/EtGr/TerFl **or** label hints
(`object_poly` / `finds_points` / `разметка` / `UTM.json` / `manifest.json`).
Layout from `docs/PREREG_w4_pixgbm.md` (box had no AWS creds → **keys unverified by live list**).

## Shared helper

Repo: `kaggle/common/koz3_s3.py` (bot6 copies verbatim). Inlined into kernel at
`scripts/build_geo_gate_pkg.py` build time.
