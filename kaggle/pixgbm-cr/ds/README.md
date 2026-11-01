# Dataset staging (coord «1» creates on Kaggle; bot-6 does not push)

| Folder | Kaggle slug | Payload |
|---|---|---|
| `pixgbm-cr-ref/` | `dmitrijkv/koz3-pixgbm-cr-ref` | v2 `result.geojson` for 200–217 **only** (missing on box) |
| `pixgbm-cr-cand/` | `dmitrijkv/koz3-pixgbm-cr-cand` | `submission_pixgbm_clean_e1g1_p4_lleaves.zip` (gitignored; on push path) |
| `pixgbm-cr-wheels/` | `dmitrijkv/koz3-pixgbm-cr-wheels` | zstandard wheels |

Also attach existing `dmitrijkv/koz3-s3-cred` (Secrets preferred). No w4-geo shards.
