# pixgbm-cr — Kaggle CPU e2e geo F1 gate (bot-6)

Private CPU kernel on **dmitrijkv**. Branch `bots/bot6-pixgbm-cleanroom`.

## Goal

REF = **v2 eb268d6** (via `fr1h.zip.bin` with **FR1h forced OFF**) vs
CAND = `submission_pixgbm_clean_e1g1_p4_lleaves.zip` on regions **200–217**:
`|dF1| ≤ 0.01` on ALL + maggr/etgr/terfl (`helpers/bot6_r3_gate.py`).

## FR1h flag

| | |
|---|---|
| Name | TOML `[post.frame_prior]` → `cfg["post"]["frame_prior"]` |
| Env var | none |
| Default in `fr1h.zip.bin` | **ON** (section present) |
| Force OFF | delete `[post.frame_prior]` + restore archive SpOr kurgan taus (`0.3645`/`0.3725`, `0.501645`/`0.645505`) |
| v2 equality | `expds/geo_pixgbm.py` sha `35e4331b…` and `models/pixgbm/all` match eb268d6; with FR1h OFF the SpOr overlay is inactive (geo path was already byte-identical to v2 even with FR1h ON per FR1H_PACK) |

## Kernel

| Field | Value |
|---|---|
| id | `dmitrijkv/koz3-bot6-pixgbm-cr-gate` |
| enable_internet | **true** |
| enable_gpu | **false** |
| code_file | `pixgbm_cr_gate.py` |

Flow: S3-fetch geo 200–217 → copy/verify `fr1h.zip.bin` SHA256 → unzip →
FR1h OFF + r3 overrides → per-region REF infer (geo_pixgbm) → CAND zip same way
(pix_clean) → GT/score → r3 gate. Checkpoints under `pixgbm_cr/ckpt_{ref,cand}/`.

## dataset_sources

```
dmitrijkv/koz3-s3-cred
dmitrijkv/koz3-fr1h-chk-20261101   # REF (exists)
dmitrijkv/koz3-pixgbm-cr-cand      # «1» creates
dmitrijkv/koz3-pixgbm-cr-wheels    # «1» creates
```

`ds/pixgbm-cr-ref/` is **unused** (superseded by fr1h-chk). Do not create `koz3-pixgbm-cr-ref`.

## Runtime (Kaggle CPU ~4 vCPU, 12h / ~20GB)

Estimate **1–3 h** for 18 regions × 2 runs (prior FR1h geo black-box ≈ 267 s on 6-cpu
image for one pass; pix_clean ens6 slower). Per-region resume keeps worst case ≪ 12 h.
Working disk: ~0.5 GB zip + ~0.6 GB unpack ×2 + ~1–2 GB geo cache ≪ 20 GB.

## Local check

```bash
python pixgbm_cr_gate.py --syntax
```

## Datasets «1» must create before kernel push

1. `dmitrijkv/koz3-pixgbm-cr-cand` ← `ds/pixgbm-cr-cand/` (zip on push path)
2. `dmitrijkv/koz3-pixgbm-cr-wheels` ← `ds/pixgbm-cr-wheels/`
3. Confirm Secrets `KOZ3_S3_*` or attachable `koz3-s3-cred`
