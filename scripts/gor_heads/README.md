# gor_heads — reproduce `gor_v3_head.npz` + `gor_dense_head.npz`

**Status:** heads were lost (Kaggle slug `dmitrijkv/koz3-b5-gordense-timing` is a phantom 403;
`/workspace/koz3_shared/bot-5/` empty; not in git; not on PC by name/size). This package
re-trains them from the original recipe.

## Recipe (confidence: high on procedure; medium on byte-identical md5)

| head | procedure | legacy md5 | legacy bytes |
|---|---|---|---|
| `gor_v3_head.npz` | `extract_feats` (DINOv2-S, 384+768 m vignettes) → PCA-64 + L2-logreg C=1 + Landauer 0.85/0.05 clean on fam10 | `acb95c52956d712a16c3dd389237a691` | 407736 |
| `gor_dense_head.npz` | `dense_feats` (GSDS 2+4, tile 518/stride 392) → `export_head` variant `all`, `centre_s` from CP_GRID (expect **250**) | `e35570e10d17edac32bea8d26fa0f5f4` | 204984 |

Sources: `FOR_BOT7_GOR_V3.md`, `scripts/gor_dense/export_head.py`, `docs/.../export_full.log`
(`centre_s=250`, inner_oof hits peak at 250), τ package lock `0.824066` (config, not in head).

**Uncertain:** re-extracted dinov2 feats may differ slightly vs original box run → md5 may not
match legacy. Acceptance = shapes/dtypes + `head_p` sanity + (for dense) recovered `centre_s=250`.

## Run on PC (2 free threads)

```bat
set DATA=F:\2\train-expds-3
set MODEL=C:\Sloi\wt_bot5_50f6f27\models\dinov2-small
set WORK=C:\Sloi\pc_jobs\bot5_gor_heads_train
set DEVICE=cpu
set THREADS=2
set OMP_NUM_THREADS=2
python scripts/gor_heads/train_heads.py
```

Resumable: deletes nothing; stages in `%WORK%/ckpt.json`; `STATUS.txt` RUNNING/DONE/FAILED.

## Box smoke (no data)

```bash
PYTHONPATH=. python scripts/gor_heads/smoke_train_heads.py
```

## Outputs

`%WORK%/heads/gor_v3_head.npz`, `gor_dense_head.npz` + sha256/md5 in `SUMMARY.json`.
Also copied to `models/gor_gate/` and `models/gor_dense/` (gitignored / not committed).
