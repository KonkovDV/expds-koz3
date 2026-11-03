# bot7 — geo-gate (ceiling + geo levers)

**Дата:** 01.11.2026 19:34 МСК (Europe/Moscow)  
**Ветка:** `bots/bot7-geo-gate` @ `573de8e5` (`573de8e5e3141bdc024c7bce6a9b45cdae0cf2c3`)  
**Worktree:** `/workspace/koz3-wt/bot7-geo-gate`  
**Package:** `/workspace/kaggle/push/geo-gate/` (не пушен)

## (1) Type shares 0.332/0.026/0.130/0.017/0.496

**Происхождение:** recovered GT counts via `lb_counts.resolve` from LB log P/R (smallest integer solution), **не** class weights.  
Log: `reports/lb/scoring_v2_eb268d6_0.41238.log` (final 0.41238).  
GT: lidar 964, aerial **75** (MATCH bot3), sat 377, ortho 50, geo 1441 / **2907**.  
Shares: 964/2907=0.3316, 75/2907=0.0258, 377/2907=0.1297, 50/2907=0.0172, 1441/2907=0.4957 ≈ **0.332/0.026/0.130/0.017/0.496**. **Вердикт: верны.**

`geogr`: в DATA_TYPE_GROUPS есть, в train GT markup **0** объектов (только maggr/etgr/terfl).

## Ceiling R→0.6 @ current P (exact F1=2PR/(P+R); P уже с 0.5·FP в логе)

| type/bucket | P | R | F1 | F1@R=0.6 | ΔF1 | ΔLB_proxy=s·ΔF1 | ΔLB_clsmap |
|---|---|---|---|---|---|---|---|
| lidar | 0.5165 | 0.5041 | 0.5102 | 0.5551 | +0.0449 | **+0.0149** | +0.0121 |
| aerial | 0.0909 | 0.0133 | 0.0233 | 0.1579 | +0.1346 | +0.0035 | **+0.0306** |
| satellite | 0.4561 | 0.2546 | 0.3268 | 0.5182 | +0.1914 | +0.0248 | **+0.0485** |
| ortho | 0.3684 | 0.1400 | 0.2029 | 0.4565 | +0.2536 | +0.0044 | +0.0124 |
| geophysics | 0.7694 | **0.6565** | 0.7085 | 0.7085 | 0 | **0** | 0 |
| size (5k,10k] | — | **0.000** | — | — | ΔR=+0.600 | n/a (no P) | share_w ≈ +0.023† |
| size >10k | — | 0.294 | — | — | ΔR=+0.306 | n/a (no P) | share_w ≈ +0.042† |

† Size: log has **recall only** (no P/GT). Assumptions: size aux↛final_score; train exclusive size×class ≈ LB; share_w = Σ (w_c/W)·ΔR·(n_{c∩b}/n_c) recall-only proxy. Optimistic_hi (all gain→povrezhdennye) much larger — не использовать как план.

**Class-map assumptions:** ΔF1_c = ΔF1_type · (n_{c∩type}/n_c) from **train** markup (LB lacks class×type); ∂S/∂F1_c = w_c/100; type aux↛final_score so proxy ≠ exact.

## (2) Geo error (local artifacts only; honest-10 OOF empty on box)

Sources: `notes/geo_probe_train.json`, `geo_tune.json`, `maggr_finds_2026-10-25.json`, `orch/geo_org_repro/R_*.json`, REVIEW_SOTA/ORG, LB log.  
`/workspace/kaggle/out/acc2-honest10/oof_honest10/` — empty STATUS only; no `geo_cur.geojson`.

| finding | numbers |
|---|---|
| LB geo | P=0.769 R=0.656 F1=0.708; center_dev global 1.13 м |
| maggr maps | 19; Rceil@2m median **0.541**; **13/19** Rceil<0.7; several Rceil=0 (Семибратнее 1/7, Гнёздово Пойма/Терраса3) |
| finds FN band | REVIEW: **809 FN** with candidate in 2–5 м (tol 2 м); oracle +0.017 F1 |
| object_poly | GT median ~6 м²; **545 FN** nearby; maggr F1_lofo 0.341; etgr 0.201; terfl 0.171 |
| maggr_finds recipe | honest Δ −0.00058 reject (notes) |
| R_maggr geo5 | mg_poly Δglob **+0.00195** (boot p_gt0 0.77); mg_pts **−0.0063** |
| R_etgr | in-sample Δglob +0.021; fam **−0.0017** |
| geogr | 0 train GT |

## Levers (flag OFF, not CLOSED)

| ID | Flag | E[ΔLB] |
|---|---|---|
| L1 | `extremum_snap_m=3.0` | ~0.001–0.0017 |
| L2 | `finds_nms_m=1.5` | ~0.0005–0.0015 |
| L3 | `post.geo_convex_hull` | +0.0017 (bot-6) |

Suite: **1001 passed, 65 skipped, 2 failed** (pre-existing `test_bot7_dfine_import` broken-vendor vs env transformers — 0 new).  
Package smoke: `GEO_GATE_SMOKE=1` → SMOKE OK.

**PR:** https://github.com/KonkovDV/expds-koz3/pull/26 (draft) tip `573de8e5`


## (3) S3 rework 01.11 19:47 MSK

| | |
|---|---|
| helper | `kaggle/common/koz3_s3.py` commit **11a47d2a** blob **7e1d0ab1…** |
| Secrets | `KOZ3_S3_ACCESS_KEY_ID` / `KOZ3_S3_SECRET_ACCESS_KEY` (= SECRET_ID/KEY like koz3-dfine-or) |
| cred fallback | `dmitrijkv/koz3-s3-cred` under `/kaggle/input/**`: json (id/key), env, ini |
| API | `s3_client`, `download`, `download_prefix`, `list_keys` (paginated); boto3+stdlib; scrub |
| slug | `dmitrijkv/koz3-bot7-geo-gate` enable_internet=true CPU |
| datasets | gt-train-v2 **verified**; xfer-s000 **verified**; s3-cred **403 unverified**; w4-geo-s000/s001 **missing→S3** |
| S3 | bucket `train-expds-3`, endpoint selcloud ru-7; prefixes 200_…217_; MagGr/EtGr/TerFl + labels |
| suite | 1016p / 65s / 2 prefails (0 new) |
| push | **не пушено**; `orch/pool_kaggle.sh dmitrijkv kernels push -p /workspace/kaggle/push/geo-gate` |
| runtime | smoke ≪1 мин; S3 ~10–40 мин; L3 post ~20–60 мин; L1/L2 re-infer 2–6 ч CPU |
