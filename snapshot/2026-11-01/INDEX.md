# Progress snapshot 2026-11-01 (Europe/Moscow)

Branch: `bots/progress-snapshot-2026-11-01` from `origin/bots/orch-protocol`.  
No LB submissions. No secrets (Kaggle tokens, S3 creds, `.env`) included.

## Headline numbers (from reports / kernel outs; not invented)

| Item | Value | Source |
|---|---|---|
| Best LB | **0.41238** (v2 `eb268d6`) | `reports/shared/bot7_geo_gate.md`, FR1h report |
| Honest-10 baseline B′ | **0.315800** | FR1h + geo sweep |
| FR1h vs B′ | Δ**+0.002905**, P**0.999**, soft-pass | `reports/repo/v2_fr1h_candidate_2026-11-01.md` |
| FR1h vs A | Δ**+0.00409**, pass | same + `fr1h-pack/w6c_gate_fr1h.json` |
| FR1h sha256 | `fe8d60b9032474760f02d4bcfa08fb43269ae3ffb872efa932d2f1d7692944b4` | `fr1h-pack/sha256.txt` |
| li-post honest LOFO (Li-layer only) | OFF F1 **0.2803** → thr_nested **0.3983**, thr_nms8 **0.3998** | `reports/shared/bot4_li_post.md`, `kaggle_out/lipost/li_post_honest.json` |
| s3-index | **2132** objects, **750** prefixes (summary header + 751 data? file has 753 lines incl. 2 hdr → ~750 prefixes), **0** fam0 hits | `kaggle_out/d_koz3-s3-index/*` |
| pixgbm-cr kernel | **ERROR** `helpers/bot6_r3_gate.py missing` | `kaggle_out/d_koz3-bot6-pixgbm-cr-gate/` |
| sat2-gate | **ERROR** `vendor/materials/compute_metrics.py missing` | `kaggle_out/sat2g/` |
| geo-gate | **COMPLETE/DONE** levers L1/L2/L3/ALL, elapsed ~95.5s (prior_only stubs) | `kaggle_out/d_koz3-bot7-geo-gate/geo_gate/` |
| sat-recall | package pushed ~20:01 MSK | `reports/shared/bot7_sat_recall.md` |

## Gates / candidates status

| Gate / candidate | Status | Key numbers |
|---|---|---|
| FR1h on eb268d6 | READY for Elena decision (no submit) | vs A +0.00409 PASS; vs B′ +0.002905 PASS_SOFT; check_submission RC=0 |
| li-post (layer) | COMPLETE on Kaggle | F1 0.2803→0.3983/0.3998; flags OFF in default.toml |
| geo-gate | COMPLETE (DONE) | expected_delta_lb stubs L1≈0.0017 L2≈0.001 L3≈0.0017; S3 pull 321 files |
| s3-index | COMPLETE | 2132 objs / ~750 prefixes / 0 fam0 |
| pixgbm-cr-gate | ERROR | missing `helpers/bot6_r3_gate.py` |
| sat2-gate | ERROR | missing `vendor/materials/compute_metrics.py` |
| sat-recall | pushed | slug `dmitrijkv/koz3-bot7-sat-recall` |

## Snapshot layout

- `reports/` — shared (`/home/box/shared/koz3/reports`), repo, workspace md/json/log
- `fr1h-pack/` — small pack artifacts (md/json/log/sha256; **no** submission zip)
- `kaggle_out/` — small kernel outs (<5MB; no train-expds-3 rasters)
- `kaggle_push/` — push package **source** only (no large bins/wheels/models)
- `KAGGLE_DATA.md` — per-account dataset matrix
- `SKIPPED_large.txt` — excluded oversized paths (if any)

## Google Drive

Folder: https://drive.google.com/drive/folders/17N5Cu1bZJ38iYu3W_nCN3zOk6R3l5riX  
Contents note: FR1h candidate split in **6 parts**, sha256 `fe8d60b9032474760f02d4bcfa08fb43269ae3ffb872efa932d2f1d7692944b4`.

## Open PRs #16–#27

| # | Title | Head | State |
|---|---|---|---|
| 16 | Aerial recall: why LB is 0.013, flags off by default | bots/bot3-aerial-recall | DRAFT open |
| 17 | fix(sp_dfine): не обнулять sp_or, если D-FINE не закрывает объекты v2 | bots/bot2-e2e-v3-prep | DRAFT open |
| 18 | Large-object recall flags, default off, honest-10 not yet measured | bots/bot4-large-objects | DRAFT open |
| 19 | bot-8: TTA и мультимасштаб спутника за флагом, D-FINE не оценён | bots/bot8-orsp-tta | DRAFT open |
| 20 | PIXGBM: переобучение с проверкой sha256 и допуск по семье | bots/bot6-pixgbm-cleanroom | DRAFT open |
| 21 | bot-5: кандидат v4, D-FINE SpOr выключен отдельно от Ae/Or | bots/bot5-candidate-v4 | DRAFT open |
| 22 | bots/bot1-speedup-1101: дисковый кэш DEM + rust_match (OFF by default) | bots/bot1-speedup-1101 | DRAFT open |
| 23 | bot-8: TTA/MS gate package koz3-tta-gate (dry-run only) | bots/bot8-tta-gate → bots/bot8-orsp-tta | DRAFT open |
| 24 | bot-5: release assemble tool (one command, draft) | bots/bot5-release-tool | DRAFT open |
| 25 | bot7: offline LB scoring.log breakdown + expected_delta_lb | bots/bot7-lb-breakdown | DRAFT open |
| 26 | bot7-geo-gate: MagGr finds extremum_snap + NMS (default OFF) | bots/bot7-geo-gate | DRAFT open |
| 27 | bot-7: sat recall levers (band-verify / agree-accept / cross-nms, OFF) | bots/bot7-sat-recall | DRAFT open |

## `bots/*` branches (origin + newly pushed)

```
origin/bots/a-within-site-rank
origin/bots/bot1-speedup-1101
origin/bots/bot2-e2e-v3-prep
origin/bots/bot3-ae2or
origin/bots/bot3-aerial-recall
origin/bots/bot4-large-objects
origin/bots/bot4-li-post
origin/bots/bot5-candidate-v4
origin/bots/bot5-release-tool
origin/bots/bot5-release-v5
origin/bots/bot6-pixgbm-cleanroom
origin/bots/bot7-geo-gate
origin/bots/bot7-lb-breakdown
origin/bots/bot7-sat-recall
origin/bots/bot8-orsp-tta
origin/bots/bot8-tta-gate
origin/bots/orch-protocol
origin/bots/w1-ae2or
origin/bots/w1-sat2-gate
origin/bots/w2-lidar-aug
origin/bots/w4-v3on
origin/bots/progress-snapshot-2026-11-01 (this branch)
```

## Kaggle datasets / kernels (CLI dump 2026-11-01; `list --mine` may omit some small/private — see also KAGGLE_DATA.md)


### dmitrijkv datasets

```
ref                                   title                                    size  lastUpdated                 downloadCount  voteCount  usabilityRating  
------------------------------------  ---------------------------------  ----------  --------------------------  -------------  ---------  ---------------  
dmitrijkv/koz3-gt-train-v2            koz3-gt-train-v2                     11465528  2026-10-29 06:12:13.160000              4          0  0.25             
dmitrijkv/koz3-xfer-s009              KOZ3 xfer shard 009                1356469680  2026-10-28 17:49:32.203000              5          0  0.125            
dmitrijkv/koz3-xfer-s007              KOZ3 xfer shard 007                1356469680  2026-10-28 17:46:50.373000             27          0  0.125            
dmitrijkv/koz3-xfer-s008              KOZ3 xfer shard 008                 579399470  2026-10-28 17:46:47.240000             11          0  0.125            
dmitrijkv/koz3-xfer-s002              KOZ3 xfer shard 002                5783543831  2026-10-28 17:46:38.210000             19          0  0.125            
dmitrijkv/koz3-xfer-s006              KOZ3 xfer shard 006                1346858029  2026-10-28 17:43:26.687000             20          0  0.125            
dmitrijkv/koz3-xfer-s005              KOZ3 xfer shard 005                 837032638  2026-10-28 17:38:48.297000             14          0  0.125            
dmitrijkv/koz3-xfer-s003              KOZ3 xfer shard 003                2001025151  2026-10-28 17:36:17.143000             48          0  0.125            
dmitrijkv/koz3-xfer-s004              KOZ3 xfer shard 004                 778189864  2026-10-28 17:34:59.277000             19          0  0.125            
dmitrijkv/koz3-xfer-s001              KOZ3 xfer shard 001                3288786683  2026-10-28 16:30:55.800000             43          0  0.125            
dmitrijkv/koz3-xfer-s000              KOZ3 xfer shard 000                   8655762  2026-10-28 16:05:42.440000              9          0  0.125            
dmitrijkv/koz3-w3-gordense-timing     koz3-w3-gordense-timing             793441595  2026-10-29 21:45:44.443000              4          0  0.375            
dmitrijkv/koz3-li-tin-59-t128-bundle  KOZ3 Li TIN 59 T128 Bundle Node 0   291348379  2026-10-28 09:22:11.993000              9          0  0.23529412       
dmitrijkv/koz3-kurgan-subclass-crops  koz3-kurgan-subclass-crops           76050073  2026-10-28 17:11:22.313000              1          0  0.125            
dmitrijkv/koz3-w3-kurgan-cpu-data     koz3-w3-kurgan-cpu-data             201105613  2026-10-29 20:08:23.890000              0          0  0.25             
dmitrijkv/koz3-w3-geo5-feat-base      koz3-w3-geo5-feat-base              574042221  2026-10-30 20:43:45.433000             54          0  0.25             
dmitrijkv/koz3-w3-wsg2-cands          koz3-w3-wsg2-cands                     318508  2026-10-30 10:28:03.807000              0          0  0.25             
dmitrijkv/koz3-w2-hsparity-code       koz3-w2-hsparity-code                  773885  2026-10-29 20:10:23.477000              0          0  0.25             
dmitrijkv/koz3-bot9-patch             koz3-bot9-patch                         42832  2026-10-28 16:38:29.470000              0          0  0.125            
dmitrijkv/koz3-w1-sat2-f14-data       koz3-w1-sat2-f14-data              2106412607  2026-11-01 11:34:38.817000              0          0  0.3125
```

### dmitrijkv kernels (page-size 50)

```
ref                                      title                          author       lastRunTime                 totalVotes  
---------------------------------------  -----------------------------  -----------  --------------------------  ----------  
dmitrijkv/koz3-tpu-smoke                 koz3-tpu-smoke                 Dmitrij-K-v  2026-11-01 11:13:20.193000           0  
dmitrijkv/koz3-w2-cnx-r50c5-f01          koz3-w2-cnx-r50c5-f01          Dmitrij-K-v  2026-10-31 05:26:55.777000           0  
dmitrijkv/koz3-w5-dino-vitb-tpu          koz3-w5-dino-vitb-tpu          Dmitrij-K-v  2026-10-31 00:04:03.647000           0  
dmitrijkv/koz3-tpu-cnx-c3sc-train        koz3-tpu-cnx-c3sc-train        Dmitrij-K-v  2026-10-30 10:17:40.760000           0  
dmitrijkv/hod-mysli-2026-tag-gpu-worker  hod-mysli-2026-tag-gpu-worker  Dmitrij-K-v  2026-10-28 07:34:35.193000           0  
dmitrijkv/koz3-bench-t4                  koz3-bench-t4                  Dmitrij-K-v  2026-10-28 18:47:20.420000           0  
dmitrijkv/koz3-bot7-sat-recall           koz3-bot7-sat-recall           Dmitrij-K-v  2026-11-01 17:01:47.250000           0  
dmitrijkv/koz3-w4-ref103                 koz3-w4-ref103                 Dmitrij-K-v  2026-11-01 16:58:18.773000           0  
dmitrijkv/koz3-bot7-geo-gate             koz3-bot7-geo-gate             Dmitrij-K-v  2026-11-01 16:54:41.990000           0  
dmitrijkv/koz3-bot6-pixgbm-cr-gate       koz3-bot6-pixgbm-cr-gate       Dmitrij-K-v  2026-11-01 17:09:15.150000           0  
dmitrijkv/koz3-s3-index                  koz3-s3-index                  Dmitrij-K-v  2026-11-01 16:54:32.490000           0  
                                         [Private Notebook]                          2010-04-01 00:00:00                  0  
dmitrijkv/koz3-w3-gordense-timing2       koz3-w3-gordense-timing2       Dmitrij-K-v  2026-11-01 16:28:26.080000           0  
dmitrijkv/koz3-w1-lidar-aug-f2           koz3-w1-lidar-aug-f2           Dmitrij-K-v  2026-11-01 15:28:48.683000           0  
dmitrijkv/koz3-w1-sat2-f14               koz3-w1-sat2-f14               Dmitrij-K-v  2026-11-01 11:37:04.907000           0  
dmitrijkv/koz3-w1-sat2-f0                koz3-w1-sat2-f0                Dmitrij-K-v  2026-11-01 09:07:43.223000           0  
dmitrijkv/koz3-sp-large-f0               koz3-sp-large-f0               Dmitrij-K-v  2026-10-31 14:00:29.917000           0  
dmitrijkv/koz3-w2-cnx-r50c5-f234         koz3-w2-cnx-r50c5-f234         Dmitrij-K-v  2026-10-31 05:53:13.373000           0  
dmitrijkv/koz3-w3-geo5-base-s34          koz3-w3-geo5-base-s34          Dmitrij-K-v  2026-10-30 20:52:40.563000           0  
dmitrijkv/koz3-w3-wsg2-cpu               koz3-w3-wsg2-cpu               Dmitrij-K-v  2026-10-30 10:28:18.400000           0  
dmitrijkv/koz3-w2-hillshade-parity-s3    koz3-w2-hillshade-parity-s3    Dmitrij-K-v  2026-10-30 05:29:39.387000           0  
dmitrijkv/koz3-w2-cnx-c3sc-s2f23         koz3-w2-cnx-c3sc-s2f23         Dmitrij-K-v  2026-10-29 20:35:23.187000           0  
dmitrijkv/koz3-w2-cnx-c3sc-s2f01         koz3-w2-cnx-c3sc-s2f01         Dmitrij-K-v  2026-10-29 20:15:46.710000           0  
dmitrijkv/koz3-w3-kurgan-cpu-seeds       koz3-w3-kurgan-cpu-seeds       Dmitrij-K-v  2026-10-30 07:14:21.353000           0  
dmitrijkv/koz3-w4-pixgbm-s3              koz3-w4-pixgbm-s3              Dmitrij-K-v  2026-10-29 19:18:42.790000           0  
dmitrijkv/koz3-cnx-li-c3sc-s1f01         koz3-cnx-li-c3sc-s1f01         Dmitrij-K-v  2026-10-29 15:03:28.160000           0  
dmitrijkv/koz3-dfine-ae-ft               koz3-dfine-ae-ft               Dmitrij-K-v  2026-10-29 13:33:15.557000           0  
dmitrijkv/koz3-bot6-pixgbm-cleanroom     koz3-bot6-pixgbm-cleanroom     Dmitrij-K-v  2026-10-29 13:37:14.390000           0  
                                         [Private Notebook]                          2010-04-01 00:00:00                  0  
dmitrijkv/koz3-cnx-li-c3sc-f4-v2         koz3-cnx-li-c3sc-f4-v2         Dmitrij-K-v  2026-10-29 13:00:23.310000           0  
dmitrijkv/koz3-dfine-or                  koz3-dfine-or                  Dmitrij-K-v  2026-10-29 12:02:37.627000           0  
dmitrijkv/koz3-dfine-ae                  koz3-dfine-ae                  Dmitrij-K-v  2026-10-29 07:28:46.957000           0  
dmitrijkv/koz3-dfine-spor-final          koz3-dfine-spor-final          Dmitrij-K-v  2026-10-29 06:02:19.220000           0  
dmitrijkv/koz3-dfine-or-prep             koz3-dfine-or-prep             Dmitrij-K-v  2026-10-29 05:04:21.430000           0  
dmitrijkv/koz3-cnx-li-c3-f4              koz3-cnx-li-c3-f4              Dmitrij-K-v  2026-10-29 04:33:01.017000           0  
dmitrijkv/koz3-cnx-li-c3sc-f23           koz3-cnx-li-c3sc-f23           Dmitrij-K-v  2026-10-29 02:59:34.727000           0  
dmitrijkv/koz3-cnx-li-c3sc-f01           koz3-cnx-li-c3sc-f01           Dmitrij-K-v  2026-10-29 01:26:01.777000           0  
dmitrijkv/koz3-cnx-li-smoke              koz3-cnx-li-smoke              Dmitrij-K-v  2026-10-29 00:56:37.767000           0  
dmitrijkv/koz3-dfine-spor-full-rest      koz3-dfine-spor-full-rest      Dmitrij-K-v  2026-10-28 22:55:53.333000           0  
dmitrijkv/koz3-dfine-spor-full           koz3-dfine-spor-full           Dmitrij-K-v  2026-10-28 19:38:46.833000           0  
dmitrijkv/koz3-cnx-li-prep               koz3-cnx-li-prep               Dmitrij-K-v  2026-10-28 19:26:32.990000           0  
dmitrijkv/koz3-dfine-spor-smoke          koz3-dfine-spor-smoke          Dmitrij-K-v  2026-10-28 19:06:58.537000           0  
dmitrijkv/koz3-nolidar-verifier          koz3-nolidar-verifier          Dmitrij-K-v  2026-10-28 18:20:27.057000           0  
dmitrijkv/koz3-kurgan-subclass           koz3-kurgan-subclass           Dmitrij-K-v  2026-10-28 17:48:46.203000           0  
dmitrijkv/koz3-gpu-s5                    koz3-gpu-s5                    Dmitrij-K-v  2026-10-28 16:39:21.567000           0  
dmitrijkv/koz3-gpu-s4                    koz3-gpu-s4                    Dmitrij-K-v  2026-10-28 16:35:17.487000           0  
dmitrijkv/koz3-gpu-s3                    koz3-gpu-s3                    Dmitrij-K-v  2026-10-28 14:54:46.567000           0  
dmitrijkv/koz3-gpu-s2                    koz3-gpu-s2                    Dmitrij-K-v  2026-10-28 13:59:02.967000           0  
dmitrijkv/koz3-gpu-s1                    koz3-gpu-s1                    Dmitrij-K-v  2026-10-28 13:59:01.360000           0  
dmitrijkv/koz3-gpu-probe                 koz3-gpu-probe                 Dmitrij-K-v  2026-10-28 13:55:27.673000           0
```

### elenakonkova datasets

```
ref                                   title                          size  lastUpdated                 downloadCount  voteCount  usabilityRating  
------------------------------------  -----------------------  ----------  --------------------------  -------------  ---------  ---------------  
elenakonkova/koz3-cnx-li-prep         koz3-cnx-li-prep          799473915  2026-10-29 17:03:35.247000              1          0  0.25             
elenakonkova/koz3-w5-cands            koz3-w5-cands               1937990  2026-10-29 20:16:36.027000              0          0  0.25             
elenakonkova/koz3-bot8-snap           koz3-bot8-snap            479642150  2026-10-29 17:07:03.580000              1          0  0.375            
elenakonkova/koz3-w2-gate             koz3-w2-gate             1653425470  2026-10-30 09:04:04.193000              5          0  0.5              
elenakonkova/koz3-w3-kurgan-cpu-data  koz3-w3-kurgan-cpu-data   201105613  2026-10-30 08:23:09.237000              0          0  0.25             
elenakonkova/koz3-w2-weights          koz3-w2-weights          1358396794  2026-10-30 09:00:19.633000              0          0  0.625            
elenakonkova/koz3-e2e-tools           koz3-e2e-tools                32879  2026-10-29 17:31:50.587000              1          0  0.25             
elenakonkova/koz3-li-aug-data         koz3-li-aug-data          210485079  2026-11-01 08:49:18.740000              0          0  0.23529412       
elenakonkova/koz3-wsr-data            koz3-wsr-data              54794887  2026-10-31 20:59:18.677000              0          0  0.375            
elenakonkova/koz3-peak-rerank-peaks   koz3-peak-rerank-peaks      2741563  2026-10-31 19:25:50.260000              0          0  0.4375           
elenakonkova/koz3-dino-gor-raw        koz3-dino-gor-raw        3709636895  2026-10-31 19:06:41.840000              0          0  0.4375           
elenakonkova/koz3-dino-gor-data       koz3-dino-gor-data        331681446  2026-10-31 18:52:17.323000              0          0  0.4375           
elenakonkova/koz3-sp-large-f0-data    koz3-sp-large-f0-data    2192843820  2026-10-31 13:24:51.953000              0          0  0.4117647        
elenakonkova/koz3-w3-gordense-timing  koz3-w3-gordense-timing   793441661  2026-10-31 08:07:14.453000              0          0  0.375            
elenakonkova/koz3-w3-geo5-feat-n40r8  koz3-w3-geo5-feat-n40r8  2315162646  2026-10-30 20:50:25.700000              0          0  0.25             
elenakonkova/koz3-w2-demfill          koz3-w2-demfill           123247104  2026-10-30 09:13:51.640000              0          0  0.5              
elenakonkova/koz3-w2-maps             koz3-w2-maps             2101891176  2026-10-30 09:01:21.073000              0          0  0.5              
elenakonkova/koz3-w2-oof              koz3-w2-oof               144806022  2026-10-30 08:59:23.810000             16          0  0.5              
elenakonkova/koz3-w1-sat2-src         koz3-w1-sat2-src             864649  2026-11-01 17:03:03.613000              0          0  0.25             
elenakonkova/koz3-s3-cred             koz3-s3-cred                    231  2026-11-01 16:53:13.427000              0          0  0.25
```

### elenakonkova kernels (page-size 50)

```
ref                                      title                       author         lastRunTime                 totalVotes  
---------------------------------------  --------------------------  -------------  --------------------------  ----------  
elenakonkova/koz3-lidar-aug-f0           koz3-lidar-aug-f0           Elena Konkova  2026-11-01 12:20:04.653000           0  
elenakonkova/koz3-dino-gor-oof5          koz3-dino-gor-oof5          Elena Konkova  2026-10-31 19:13:48.640000           0  
elenakonkova/koz3-w5-dino-vitb-tpu       koz3-w5-dino-vitb-tpu       Elena Konkova  2026-10-31 00:04:01.070000           0  
elenakonkova/koz3-acc2-tpu-smoke         koz3-acc2-tpu-smoke         Elena Konkova  2026-10-30 10:17:39.183000           0  
elenakonkova/koz3-acc2-gpu-smoke         koz3-acc2-gpu-smoke         Elena Konkova  2026-10-29 16:45:13.417000           0  
                                         [Private Notebook]                         2010-04-01 00:00:00                  0  
elenakonkova/koz3-bot7-sat-recall        koz3-bot7-sat-recall        Elena Konkova  2026-11-01 17:15:22.803000           0  
elenakonkova/koz3-bot7-geo-gate          koz3-bot7-geo-gate          Elena Konkova  2026-11-01 17:15:17.627000           0  
elenakonkova/koz3-w4-li-post-full        koz3-w4-li-post-full        Elena Konkova  2026-11-01 17:04:39.537000           0  
elenakonkova/koz3-s3-index               koz3-s3-index               Elena Konkova  2026-11-01 16:39:31.537000           0  
elenakonkova/koz3-w4-li-post-gate        koz3-w4-li-post-gate        Elena Konkova  2026-11-01 16:37:55.457000           0  
                                         [Private Notebook]                         2010-04-01 00:00:00                  0  
elenakonkova/koz3-w1-sat2-gate           koz3-w1-sat2-gate           Elena Konkova  2026-11-01 17:10:17.343000           0  
elenakonkova/koz3-wsr-f0                 koz3-wsr-f0                 Elena Konkova  2026-10-31 20:59:43.443000           0  
elenakonkova/koz3-peak-rerank-f0         koz3-peak-rerank-f0         Elena Konkova  2026-10-31 19:30:00.563000           0  
elenakonkova/koz3-dino-gor-f0            koz3-dino-gor-f0            Elena Konkova  2026-10-31 18:54:16.107000           0  
elenakonkova/koz3-w3-gordense-timing-k   koz3-w3-gordense-timing-k   Elena Konkova  2026-10-31 08:37:01.780000           0  
elenakonkova/koz3-w2-cnx-r50c5-f01       koz3-w2-cnx-r50c5-f01       Elena Konkova  2026-10-31 05:41:54.510000           0  
elenakonkova/koz3-w3-kurgan-cpu-seeds-b  koz3-w3-kurgan-cpu-seeds-b  Elena Konkova  2026-10-30 10:19:17.063000           0  
elenakonkova/koz3-w3-kurgan-cpu-seeds    koz3-w3-kurgan-cpu-seeds    Elena Konkova  2026-10-30 08:23:31.903000           0  
elenakonkova/koz3-w5-cloudfeat           koz3-w5-cloudfeat           Elena Konkova  2026-10-30 08:07:53.740000           0  
elenakonkova/koz3-w5-dino-emb-gpu        koz3-w5-dino-emb-gpu        Elena Konkova  2026-10-30 03:04:46.857000           0  
elenakonkova/koz3-w2-lihm-c3sc-f01234    koz3-w2-lihm-c3sc-f01234    Elena Konkova  2026-10-30 01:04:51.153000           0  
elenakonkova/koz3-w2-cnx-c3sc-s3f4       koz3-w2-cnx-c3sc-s3f4       Elena Konkova  2026-10-30 00:35:42.990000           0  
elenakonkova/koz3-w5-dino-emb            koz3-w5-dino-emb            Elena Konkova  2026-10-29 20:18:07.480000           0  
elenakonkova/koz3-w1-dfine-or-final      koz3-w1-dfine-or-final      Elena Konkova  2026-10-29 19:38:36.477000           0  
elenakonkova/koz3-w1-dfine-ae-final      koz3-w1-dfine-ae-final      Elena Konkova  2026-10-29 19:23:22.370000           0  
elenakonkova/koz3-w2-cnx-c3sc-s2f4       koz3-w2-cnx-c3sc-s2f4       Elena Konkova  2026-10-29 22:04:12.197000           0  
elenakonkova/koz3-cnx-li-c3sc-s1f23      koz3-cnx-li-c3sc-s1f23      Elena Konkova  2026-10-29 17:06:01.057000           0  
elenakonkova/koz3-bot8-oof-honest10      koz3-bot8-oof-honest10      Elena Konkova  2026-10-29 17:23:34.260000           0
```

### usertest1234 datasets

```
ref                                      title                                    size  lastUpdated                 downloadCount  voteCount  usabilityRating  
---------------------------------------  ---------------------------------  ----------  --------------------------  -------------  ---------  ---------------  
usertest1234/koz3-w3-kurgan-cpu-data     koz3-w3-kurgan-cpu-data             201105613  2026-10-29 19:34:05.750000              0          0  0.25             
usertest1234/koz3-w4-v2clean-check       koz3-w4-v2clean-check               526506473  2026-10-29 22:52:51.523000              0          0  0.25             
usertest1234/koz3-li-tin-59-t128-bundle  KOZ3 Li TIN 59 T128 Bundle Node 1   291348379  2026-10-28 09:22:41.877000              0          0  0.23529412       
usertest1234/koz3-w3-geo5-feat-n40r8     koz3-w3-geo5-feat-n40r8            2315162646  2026-10-30 20:42:07.833000              0          0  0.25
```

### usertest1234 kernels (page-size 50)

```
ref                                         title                          author         lastRunTime                 totalVotes  
------------------------------------------  -----------------------------  -------------  --------------------------  ----------  
usertest1234/hod-mysli-2026-tag-gpu-worker  hod-mysli-2026-tag-gpu-worker  User Test1234  2026-10-28 07:34:39.043000           0  
usertest1234/koz3-w4-fr1h-check             koz3-w4-fr1h-check             User Test1234  2026-11-01 16:21:32.013000           0  
usertest1234/koz3-w3-kurgan-cpu-seeds-b     koz3-w3-kurgan-cpu-seeds-b     User Test1234  2026-10-30 10:42:00.430000           0  
usertest1234/koz3-w4-v2cleanchk-k           koz3-w4-v2cleanchk-k           User Test1234  2026-10-30 04:37:58.037000           0  
usertest1234/koz3-w3-kurgan-cpu-seeds       koz3-w3-kurgan-cpu-seeds       User Test1234  2026-10-30 05:08:37.800000           0  
usertest1234/koz3-gpu-probe                 koz3-gpu-probe                 User Test1234  2026-10-28 13:57:50.773000           0  
usertest1234/koz3-li-tin-59-t128-dual-t4    koz3-li-tin-59-t128-dual-t4    User Test1234  2026-10-28 09:28:21.623000           0
```

### aaaadg4gg4g4g datasets

```
ref                                       title                                    size  lastUpdated                 downloadCount  voteCount  usabilityRating  
----------------------------------------  ---------------------------------  ----------  --------------------------  -------------  ---------  ---------------  
aaaadg4gg4g4g/koz3-w3-kurgan-cpu-data     koz3-w3-kurgan-cpu-data             201105613  2026-10-29 19:34:26.450000              0          0  0.25             
aaaadg4gg4g4g/koz3-li-tin-59-t128-bundle  KOZ3 Li TIN 59 T128 Bundle Node 3   291348379  2026-10-28 09:23:35.923000              0          0  0.23529412       
aaaadg4gg4g4g/koz3-w3-geo5-feat-n40r8     koz3-w3-geo5-feat-n40r8            2315162646  2026-10-30 20:42:06.187000              0          0  0.25
```

### aaaadg4gg4g4g kernels (page-size 50)

```
ref                                        title                        author          lastRunTime                 totalVotes  
-----------------------------------------  ---------------------------  --------------  --------------------------  ----------  
aaaadg4gg4g4g/koz3-w4-fr1h-gate            koz3-w4-fr1h-gate            AA AAdg4gg4g4g  2026-11-01 16:21:32.317000           0  
aaaadg4gg4g4g/koz3-w3-kurgan-cpu-seeds-b   koz3-w3-kurgan-cpu-seeds-b   AA AAdg4gg4g4g  2026-10-30 10:42:04.423000           0  
aaaadg4gg4g4g/koz3-w3-kurgan-cpu-seeds     koz3-w3-kurgan-cpu-seeds     AA AAdg4gg4g4g  2026-10-30 07:51:04.310000           0  
aaaadg4gg4g4g/koz3-gpu-probe               koz3-gpu-probe               AA AAdg4gg4g4g  2026-10-28 13:57:55.660000           0  
aaaadg4gg4g4g/koz3-li-tin-59-t128-dual-t4  koz3-li-tin-59-t128-dual-t4  AA AAdg4gg4g4g  2026-10-28 09:28:26.287000           0
```

### aaaa0000004 datasets

```
ref                                     title                                    size  lastUpdated                 downloadCount  voteCount  usabilityRating  
--------------------------------------  ---------------------------------  ----------  --------------------------  -------------  ---------  ---------------  
aaaa0000004/koz3-w4-v2clean-check       koz3-w4-v2clean-check               526506643  2026-10-30 06:20:30.640000              1          0  0.25             
aaaa0000004/koz3-w4-v3a-min-check       koz3-w4-v3a-min-check               526506401  2026-10-30 10:34:19.950000              0          0  0.25             
aaaa0000004/koz3-w4-wheels              koz3-w4-wheels                         851466  2026-10-30 05:54:44.137000              1          0  0.25             
aaaa0000004/koz3-w3-geo5-feat-base      koz3-w3-geo5-feat-base              574042221  2026-10-30 20:43:41.913000              0          0  0.25             
aaaa0000004/koz3-w4-wheels-v3           koz3-w4-wheels-v3                    27005800  2026-10-30 12:59:36.107000              0          0  0.25             
aaaa0000004/koz3-w4-v3on-check          koz3-w4-v3on-check                  661130738  2026-10-30 12:59:10.917000              0          0  0.25             
aaaa0000004/koz3-w2-gaudit-las          koz3-w2-gaudit-las                 3466757301  2026-10-30 05:30:41.177000              0          0  0.25             
aaaa0000004/koz3-w3-kurgan-cpu-data     koz3-w3-kurgan-cpu-data             201105613  2026-10-29 19:34:16.110000              0          0  0.25             
aaaa0000004/koz3-li-tin-59-t128-bundle  KOZ3 Li TIN 59 T128 Bundle Node 2   291348379  2026-10-28 09:23:08.070000              0          0  0.23529412       
aaaa0000004/koz3-w3-geo5-feat-n40r8     koz3-w3-geo5-feat-n40r8            2315162646  2026-10-30 21:41:45.533000              0          0  0.25             
aaaa0000004/koz3-w4-dfine-ref103        koz3-w4-dfine-ref103                 26358369  2026-11-01 17:00:26.013000              0          0  0.25
```

### aaaa0000004 kernels (page-size 50)

```
ref                                      title                        author        lastRunTime                 totalVotes  
---------------------------------------  ---------------------------  ------------  --------------------------  ----------  
aaaa0000004/koz3-w4-fr1h-geo             koz3-w4-fr1h-geo             AA AA0000004  2026-11-01 16:21:31.147000           0  
aaaa0000004/koz3-w2-cnx-r50c5-f01        koz3-w2-cnx-r50c5-f01        AA AA0000004  2026-10-31 00:12:23.513000           0  
aaaa0000004/koz3-w3-geo5-base-s12        koz3-w3-geo5-base-s12        AA AA0000004  2026-10-30 20:52:38.703000           0  
aaaa0000004/koz3-w4-v3onchk              koz3-w4-v3onchk              AA AA0000004  2026-11-01 17:01:59.017000           0  
aaaa0000004/koz3-w4-v3amin-chk           koz3-w4-v3amin-chk           AA AA0000004  2026-10-30 10:34:43.273000           0  
aaaa0000004/koz3-w4-v2cleanchk2          koz3-w4-v2cleanchk2          AA AA0000004  2026-10-30 06:25:06.660000           0  
aaaa0000004/koz3-w2-ground-clip-audit    koz3-w2-ground-clip-audit    AA AA0000004  2026-10-30 05:33:52.397000           0  
aaaa0000004/koz3-w3-kurgan-cpu-seeds     koz3-w3-kurgan-cpu-seeds     AA AA0000004  2026-10-30 07:51:01.580000           0  
aaaa0000004/koz3-gpu-probe               koz3-gpu-probe               AA AA0000004  2026-10-28 13:57:52.897000           0  
aaaa0000004/koz3-li-tin-59-t128-dual-t4  koz3-li-tin-59-t128-dual-t4  AA AA0000004  2026-10-28 09:28:23.007000           0
```

### Known extras not always shown by default `datasets list --mine`

- `dmitrijkv/koz3-fr1h-chk-20261101`, `dmitrijkv/koz3-fr1h-wheels`, `dmitrijkv/koz3-s3-cred`
- `elenakonkova/koz3-s3-cred`; created this snapshot: `elenakonkova/koz3-gt-train-v2`, `elenakonkova/koz3-fr1h-chk-20261101`
- CPU copies created this snapshot: `*/koz3-gt-train-v2` on usertest1234, aaaadg4gg4g4g, aaaa0000004

## Live ops notes (01.11 evening MSK)
- S3 cred private: `dmitrijkv/koz3-s3-cred`, `elenakonkova/koz3-s3-cred` (not on CPU).
- dmitrijkv cannot read `elenakonkova/koz3-w2-gate`.
- Real bot7 geo-gate + sat-recall running on elenakonkova since 20:15.
- li-post-full COMPLETE on elenakonkova.
- large-gate push rejected 400 (≈3.6 MB script).
- See `KAGGLE_DATA.md` for dataset matrix.
