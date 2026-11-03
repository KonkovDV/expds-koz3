# Infra
- Kaggle accounts: GPU dmitrijkv, elenakonkova (2×T4, ~30 GPUh/week each, max 2 GPU sessions); CPU usertest1234, aaaa0000004, aaaadg4gg4g4g (no internet/S3).
- Tokens: box ~/.secrets/kpool/<user>; use `KAGGLE_API_TOKEN=$(cat ~/.secrets/kpool/<user>) kaggle ...`. Never print.
- Private cred datasets: dmitrijkv/koz3-s3-cred, elenakonkova/koz3-s3-cred (GPU accounts only).
- dmitrijkv can't read elenakonkova/koz3-w2-gate → gates needing it run on elenakonkova.
- Script kernels upload only the code file → embed helpers (gzip+base64) or use a dataset; keep script <1MB.
- Kaggle auto-extracts zips in datasets → upload as .zip.bin.
- Leases: `orch/lease.sh acquire <res> <owner> <ttl_h> --kernel acc/slug --note "..."`.
- Push packages: /workspace/kaggle/push/*; outputs: /workspace/kaggle_out/*.
- Elena (elenakonkova): the only one who submits to LB.
- Bot roster: 1 orchestrator; bots 4,6,7 active (economy); 2,3,5,8 paused.
- Handoff dataset on each account: <user>/koz3-handoff.

## Datasets & kernels per account (2026-11-01)
### dmitrijkv
```
ref                                   title                                    size  lastU
------------------------------------  ---------------------------------  ----------  -----
dmitrijkv/koz3-gt-train-v2            koz3-gt-train-v2                     11465528  2026-
dmitrijkv/koz3-xfer-s009              KOZ3 xfer shard 009                1356469680  2026-
dmitrijkv/koz3-xfer-s007              KOZ3 xfer shard 007                1356469680  2026-
dmitrijkv/koz3-xfer-s008              KOZ3 xfer shard 008                 579399470  2026-
dmitrijkv/koz3-xfer-s002              KOZ3 xfer shard 002                5783543831  2026-
dmitrijkv/koz3-xfer-s006              KOZ3 xfer shard 006                1346858029  2026-
dmitrijkv/koz3-xfer-s005              KOZ3 xfer shard 005                 837032638  2026-
dmitrijkv/koz3-xfer-s003              KOZ3 xfer shard 003                2001025151  2026-
dmitrijkv/koz3-xfer-s004              KOZ3 xfer shard 004                 778189864  2026-
dmitrijkv/koz3-xfer-s001              KOZ3 xfer shard 001                3288786683  2026-
dmitrijkv/koz3-xfer-s000              KOZ3 xfer shard 000                   8655762  2026-
dmitrijkv/koz3-w3-gordense-timing     koz3-w3-gordense-timing             793441595  2026-
dmitrijkv/koz3-li-tin-59-t128-bundle  KOZ3 Li TIN 59 T128 Bundle Node 0   291348379  2026-
dmitrijkv/koz3-kurgan-subclass-crops  koz3-kurgan-subclass-crops           76050073  2026-
dmitrijkv/koz3-w3-kurgan-cpu-data     koz3-w3-kurgan-cpu-data             201105613  2026-
dmitrijkv/koz3-w3-geo5-feat-base      koz3-w3-geo5-feat-base              574042221  2026-
dmitrijkv/koz3-w3-wsg2-cands          koz3-w3-wsg2-cands                     318508  2026-
dmitrijkv/koz3-w2-hsparity-code       koz3-w2-hsparity-code                  773885  2026-
dmitrijkv/koz3-bot9-patch             koz3-bot9-patch                         42832  2026-
dmitrijkv/koz3-w1-sat2-f14-data       koz3-w1-sat2-f14-data              2106412607  2026-
ref                                          title                               author   
-------------------------------------------  ----------------------------------  ---------
dmitrijkv/koz3-tpu-smoke                     koz3-tpu-smoke                      Dmitrij-K
dmitrijkv/koz3-w2-cnx-r50c5-f01              koz3-w2-cnx-r50c5-f01               Dmitrij-K
dmitrijkv/koz3-w5-dino-vitb-tpu              koz3-w5-dino-vitb-tpu               Dmitrij-K
dmitrijkv/koz3-tpu-cnx-c3sc-train            koz3-tpu-cnx-c3sc-train             Dmitrij-K
dmitrijkv/hod-mysli-2026-tag-gpu-worker      hod-mysli-2026-tag-gpu-worker       Dmitrij-K
dmitrijkv/koz3-bench-t4                      koz3-bench-t4                       Dmitrij-K
dmitrijkv/koz3-bot7-sat-recall               koz3-bot7-sat-recall                Dmitrij-K
dmitrijkv/koz3-w4-ref103                     koz3-w4-ref103                      Dmitrij-K
dmitrijkv/koz3-bot7-geo-gate                 koz3-bot7-geo-gate                  Dmitrij-K
dmitrijkv/koz3-bot6-pixgbm-cr-gate           koz3-bot6-pixgbm-cr-gate            Dmitrij-K
dmitrijkv/koz3-s3-index                      koz3-s3-index                       Dmitrij-K
dmitrijkv/koz3-w1-ae-dfine-gate              koz3-w1-ae-dfine-gate               Dmitrij-K
dmitrijkv/koz3-w3-gordense-timing2           koz3-w3-gordense-timing2            Dmitrij-K
dmitrijkv/koz3-w1-lidar-aug-f2               koz3-w1-lidar-aug-f2                Dmitrij-K
dmitrijkv/koz3-w1-sat2-f14                   koz3-w1-sat2-f14                    Dmitrij-K
dmitrijkv/koz3-w1-sat2-f0                    koz3-w1-sat2-f0                     Dmitrij-K
dmitrijkv/koz3-sp-large-f0                   koz3-sp-large-f0                    Dmitrij-K
dmitrijkv/koz3-w2-cnx-r50c5-f234             koz3-w2-cnx-r50c5-f234              Dmitrij-K
dmitrijkv/koz3-w3-geo5-base-s34              koz3-w3-geo5-base-s34               Dmitrij-K
dmitrijkv/koz3-w3-wsg2-cpu                   koz3-w3-wsg2-cpu                    Dmitrij-K
dmitrijkv/koz3-w2-hillshade-parity-s3        koz3-w2-hillshade-parity-s3         Dmitrij-K
dmitrijkv/koz3-w2-cnx-c3sc-s2f23             koz3-w2-cnx-c3sc-s2f23              Dmitrij-K
dmitrijkv/koz3-w2-cnx-c3sc-s2f01             koz3-w2-cnx-c3sc-s2f01              Dmitrij-K
dmitrijkv/koz3-w3-kurgan-cpu-seeds           koz3-w3-kurgan-cpu-seeds            Dmitrij-K
dmitrijkv/koz3-w4-pixgbm-s3                  koz3-w4-pixgbm-s3                   Dmitrij-K
dmitrijkv/koz3-cnx-li-c3sc-s1f01             koz3-cnx-li-c3sc-s1f01              Dmitrij-K
dmitrijkv/koz3-dfine-ae-ft                   koz3-dfine-ae-ft                    Dmitrij-K
dmitrijkv/koz3-bot6-pixgbm-cleanroom         koz3-bot6-pixgbm-cleanroom          Dmitrij-K
                                             [Private Notebook]                           
dmitrijkv/koz3-cnx-li-c3sc-f4-v2             koz3-cnx-li-c3sc-f4-v2              Dmitrij-K
dmitrijkv/koz3-dfine-or                      koz3-dfine-or                       Dmitrij-K
dmitrijkv/koz3-dfine-ae                      koz3-dfine-ae                       Dmitrij-K
dmitrijkv/koz3-dfine-spor-final              koz3-dfine-spor-final               Dmitrij-K
dmitrijkv/koz3-dfine-or-prep                 koz3-dfine-or-prep                  Dmitrij-K
dmitrijkv/koz3-cnx-li-c3-f4                  koz3-cnx-li-c3-f4                   Dmitrij-K
dmitrijkv/koz3-cnx-li-c3sc-f23               koz3-cnx-li-c3sc-f23                Dmitrij-K
dmitrijkv/koz3-cnx-li-c3sc-f01               koz3-cnx-li-c3sc-f01                Dmitrij-K
dmitrijkv/koz3-cnx-li-smoke                  koz3-cnx-li-smoke                   Dmitrij-K
dmitrijkv/koz3-dfine-spor-full-rest          koz3-dfine-spor-full-rest           Dmitrij-K
dmitrijkv/koz3-dfine-spor-full               koz3-dfine-spor-full                Dmitrij-K
dmitrijkv/koz3-cnx-li-prep                   koz3-cnx-li-prep                    Dmitrij-K
dmitrijkv/koz3-dfine-spor-smoke              koz3-dfine-spor-smoke               Dmitrij-K
dmitrijkv/koz3-nolidar-verifier              koz3-nolidar-verifier               Dmitrij-K
dmitrijkv/koz3-kurgan-subclass               koz3-kurgan-subclass                Dmitrij-K
dmitrijkv/koz3-gpu-s5                        koz3-gpu-s5                         Dmitrij-K
dmitrijkv/koz3-gpu-s4                        koz3-gpu-s4                         Dmitrij-K
dmitrijkv/koz3-gpu-s3                        koz3-gpu-s3                         Dmitrij-K
dmitrijkv/koz3-gpu-s2                        koz3-gpu-s2                         Dmitrij-K
dmitrijkv/koz3-gpu-s1                        koz3-gpu-s1                         Dmitrij-K
dmitrijkv/koz3-gpu-probe                     koz3-gpu-probe                      Dmitrij-K
dmitrijkv/koz3-li-tin-59-t128-dual-t4        koz3-li-tin-59-t128-dual-t4         Dmitrij-K
dmitrijkv/exercise-arithmetic-and-variables  Exercise: Arithmetic and Variables  Dmitrij-K
```
### elenakonkova
```
ref                                   title                          size  lastUpdated    
------------------------------------  -----------------------  ----------  ---------------
elenakonkova/koz3-cnx-li-prep         koz3-cnx-li-prep          799473915  2026-10-29 17:0
elenakonkova/koz3-w5-cands            koz3-w5-cands               1937990  2026-10-29 20:1
elenakonkova/koz3-bot8-snap           koz3-bot8-snap            479642150  2026-10-29 17:0
elenakonkova/koz3-w2-gate             koz3-w2-gate             1653425470  2026-10-30 09:0
elenakonkova/koz3-w3-kurgan-cpu-data  koz3-w3-kurgan-cpu-data   201105613  2026-10-30 08:2
elenakonkova/koz3-w2-weights          koz3-w2-weights          1358396794  2026-10-30 09:0
elenakonkova/koz3-e2e-tools           koz3-e2e-tools                32879  2026-10-29 17:3
elenakonkova/koz3-li-aug-data         koz3-li-aug-data          210485079  2026-11-01 08:4
elenakonkova/koz3-wsr-data            koz3-wsr-data              54794887  2026-10-31 20:5
elenakonkova/koz3-peak-rerank-peaks   koz3-peak-rerank-peaks      2741563  2026-10-31 19:2
elenakonkova/koz3-dino-gor-raw        koz3-dino-gor-raw        3709636895  2026-10-31 19:0
elenakonkova/koz3-dino-gor-data       koz3-dino-gor-data        331681446  2026-10-31 18:5
elenakonkova/koz3-sp-large-f0-data    koz3-sp-large-f0-data    2192843820  2026-10-31 13:2
elenakonkova/koz3-w3-gordense-timing  koz3-w3-gordense-timing   793441661  2026-10-31 08:0
elenakonkova/koz3-w3-geo5-feat-n40r8  koz3-w3-geo5-feat-n40r8  2315162646  2026-10-30 20:5
elenakonkova/koz3-w2-demfill          koz3-w2-demfill           123247104  2026-10-30 09:1
elenakonkova/koz3-w2-maps             koz3-w2-maps             2101891176  2026-10-30 09:0
elenakonkova/koz3-w2-oof              koz3-w2-oof               144806022  2026-10-30 08:5
elenakonkova/koz3-w4-large-gate-code  koz3-w4-large-gate-code     2678241  2026-11-01 17:2
elenakonkova/koz3-w3-geo5-feat-base   koz3-w3-geo5-feat-base    574042221  2026-11-01 17:1
ref                                      title                       author         lastRu
---------------------------------------  --------------------------  -------------  ------
elenakonkova/koz3-lidar-aug-f0           koz3-lidar-aug-f0           Elena Konkova  2026-1
elenakonkova/koz3-dino-gor-oof5          koz3-dino-gor-oof5          Elena Konkova  2026-1
elenakonkova/koz3-w5-dino-vitb-tpu       koz3-w5-dino-vitb-tpu       Elena Konkova  2026-1
elenakonkova/koz3-acc2-tpu-smoke         koz3-acc2-tpu-smoke         Elena Konkova  2026-1
elenakonkova/koz3-acc2-gpu-smoke         koz3-acc2-gpu-smoke         Elena Konkova  2026-1
elenakonkova/koz3-w4-large-gate          koz3-w4-large-gate          Elena Konkova  2026-1
elenakonkova/koz3-bot7-sat-recall        koz3-bot7-sat-recall        Elena Konkova  2026-1
elenakonkova/koz3-bot7-geo-gate          koz3-bot7-geo-gate          Elena Konkova  2026-1
elenakonkova/koz3-w4-li-post-full        koz3-w4-li-post-full        Elena Konkova  2026-1
elenakonkova/koz3-s3-index               koz3-s3-index               Elena Konkova  2026-1
elenakonkova/koz3-w4-li-post-gate        koz3-w4-li-post-gate        Elena Konkova  2026-1
                                         [Private Notebook]                         2010-0
elenakonkova/koz3-w1-sat2-gate           koz3-w1-sat2-gate           Elena Konkova  2026-1
elenakonkova/koz3-wsr-f0                 koz3-wsr-f0                 Elena Konkova  2026-1
elenakonkova/koz3-peak-rerank-f0         koz3-peak-rerank-f0         Elena Konkova  2026-1
elenakonkova/koz3-dino-gor-f0            koz3-dino-gor-f0            Elena Konkova  2026-1
elenakonkova/koz3-w3-gordense-timing-k   koz3-w3-gordense-timing-k   Elena Konkova  2026-1
elenakonkova/koz3-w2-cnx-r50c5-f01       koz3-w2-cnx-r50c5-f01       Elena Konkova  2026-1
elenakonkova/koz3-w3-kurgan-cpu-seeds-b  koz3-w3-kurgan-cpu-seeds-b  Elena Konkova  2026-1
elenakonkova/koz3-w3-kurgan-cpu-seeds    koz3-w3-kurgan-cpu-seeds    Elena Konkova  2026-1
elenakonkova/koz3-w5-cloudfeat           koz3-w5-cloudfeat           Elena Konkova  2026-1
elenakonkova/koz3-w5-dino-emb-gpu        koz3-w5-dino-emb-gpu        Elena Konkova  2026-1
elenakonkova/koz3-w2-lihm-c3sc-f01234    koz3-w2-lihm-c3sc-f01234    Elena Konkova  2026-1
elenakonkova/koz3-w2-cnx-c3sc-s3f4       koz3-w2-cnx-c3sc-s3f4       Elena Konkova  2026-1
elenakonkova/koz3-w5-dino-emb            koz3-w5-dino-emb            Elena Konkova  2026-1
elenakonkova/koz3-w1-dfine-or-final      koz3-w1-dfine-or-final      Elena Konkova  2026-1
elenakonkova/koz3-w1-dfine-ae-final      koz3-w1-dfine-ae-final      Elena Konkova  2026-1
elenakonkova/koz3-w2-cnx-c3sc-s2f4       koz3-w2-cnx-c3sc-s2f4       Elena Konkova  2026-1
elenakonkova/koz3-cnx-li-c3sc-s1f23      koz3-cnx-li-c3sc-s1f23      Elena Konkova  2026-1
elenakonkova/koz3-bot8-oof-honest10      koz3-bot8-oof-honest10      Elena Konkova  2026-1
```
### usertest1234
```
ref                                      title                                    size  la
---------------------------------------  ---------------------------------  ----------  --
usertest1234/koz3-w3-kurgan-cpu-data     koz3-w3-kurgan-cpu-data             201105613  20
usertest1234/koz3-w4-v2clean-check       koz3-w4-v2clean-check               526506473  20
usertest1234/koz3-li-tin-59-t128-bundle  KOZ3 Li TIN 59 T128 Bundle Node 1   291348379  20
usertest1234/koz3-w3-geo5-feat-n40r8     koz3-w3-geo5-feat-n40r8            2315162646  20
usertest1234/koz3-w3-geo5-feat-base      koz3-w3-geo5-feat-base              574042221  20
usertest1234/koz3-gt-train-v2            koz3-gt-train-v2                     11465528  20
ref                                         title                          author         
------------------------------------------  -----------------------------  -------------  
usertest1234/hod-mysli-2026-tag-gpu-worker  hod-mysli-2026-tag-gpu-worker  User Test1234  
usertest1234/koz3-w4-fr1h-check             koz3-w4-fr1h-check             User Test1234  
usertest1234/koz3-w3-kurgan-cpu-seeds-b     koz3-w3-kurgan-cpu-seeds-b     User Test1234  
usertest1234/koz3-w4-v2cleanchk-k           koz3-w4-v2cleanchk-k           User Test1234  
usertest1234/koz3-w3-kurgan-cpu-seeds       koz3-w3-kurgan-cpu-seeds       User Test1234  
usertest1234/koz3-gpu-probe                 koz3-gpu-probe                 User Test1234  
usertest1234/koz3-li-tin-59-t128-dual-t4    koz3-li-tin-59-t128-dual-t4    User Test1234  
```
### aaaadg4gg4g4g
```
ref                                       title                                    size  l
----------------------------------------  ---------------------------------  ----------  -
aaaadg4gg4g4g/koz3-w3-kurgan-cpu-data     koz3-w3-kurgan-cpu-data             201105613  2
aaaadg4gg4g4g/koz3-li-tin-59-t128-bundle  KOZ3 Li TIN 59 T128 Bundle Node 3   291348379  2
aaaadg4gg4g4g/koz3-w3-geo5-feat-n40r8     koz3-w3-geo5-feat-n40r8            2315162646  2
aaaadg4gg4g4g/koz3-gt-train-v2            koz3-gt-train-v2                     11465528  2
ref                                        title                        author          la
-----------------------------------------  ---------------------------  --------------  --
aaaadg4gg4g4g/koz3-w4-fr1h-gate            koz3-w4-fr1h-gate            AA AAdg4gg4g4g  20
aaaadg4gg4g4g/koz3-w3-kurgan-cpu-seeds-b   koz3-w3-kurgan-cpu-seeds-b   AA AAdg4gg4g4g  20
aaaadg4gg4g4g/koz3-w3-kurgan-cpu-seeds     koz3-w3-kurgan-cpu-seeds     AA AAdg4gg4g4g  20
aaaadg4gg4g4g/koz3-gpu-probe               koz3-gpu-probe               AA AAdg4gg4g4g  20
aaaadg4gg4g4g/koz3-li-tin-59-t128-dual-t4  koz3-li-tin-59-t128-dual-t4  AA AAdg4gg4g4g  20
```
### aaaa0000004
```
ref                                     title                                    size  las
--------------------------------------  ---------------------------------  ----------  ---
aaaa0000004/koz3-w4-v2clean-check       koz3-w4-v2clean-check               526506643  202
aaaa0000004/koz3-w4-v3a-min-check       koz3-w4-v3a-min-check               526506401  202
aaaa0000004/koz3-w4-wheels              koz3-w4-wheels                         851466  202
aaaa0000004/koz3-w3-geo5-feat-base      koz3-w3-geo5-feat-base              574042221  202
aaaa0000004/koz3-w4-wheels-v3           koz3-w4-wheels-v3                    27005800  202
aaaa0000004/koz3-w4-v3on-check          koz3-w4-v3on-check                  661130738  202
aaaa0000004/koz3-w2-gaudit-las          koz3-w2-gaudit-las                 3466757301  202
aaaa0000004/koz3-w3-kurgan-cpu-data     koz3-w3-kurgan-cpu-data             201105613  202
aaaa0000004/koz3-li-tin-59-t128-bundle  KOZ3 Li TIN 59 T128 Bundle Node 2   291348379  202
aaaa0000004/koz3-w3-geo5-feat-n40r8     koz3-w3-geo5-feat-n40r8            2315162646  202
aaaa0000004/koz3-gt-train-v2            koz3-gt-train-v2                     11465528  202
aaaa0000004/koz3-w4-dfine-ref103        koz3-w4-dfine-ref103                 26358369  202
ref                                      title                        author        lastRu
---------------------------------------  ---------------------------  ------------  ------
aaaa0000004/koz3-w4-fr1h-geo             koz3-w4-fr1h-geo             AA AA0000004  2026-1
aaaa0000004/koz3-w2-cnx-r50c5-f01        koz3-w2-cnx-r50c5-f01        AA AA0000004  2026-1
aaaa0000004/koz3-w3-geo5-base-s12        koz3-w3-geo5-base-s12        AA AA0000004  2026-1
aaaa0000004/koz3-w4-v3onchk              koz3-w4-v3onchk              AA AA0000004  2026-1
aaaa0000004/koz3-w4-v3amin-chk           koz3-w4-v3amin-chk           AA AA0000004  2026-1
aaaa0000004/koz3-w4-v2cleanchk2          koz3-w4-v2cleanchk2          AA AA0000004  2026-1
aaaa0000004/koz3-w2-ground-clip-audit    koz3-w2-ground-clip-audit    AA AA0000004  2026-1
aaaa0000004/koz3-w3-kurgan-cpu-seeds     koz3-w3-kurgan-cpu-seeds     AA AA0000004  2026-1
aaaa0000004/koz3-gpu-probe               koz3-gpu-probe               AA AA0000004  2026-1
aaaa0000004/koz3-li-tin-59-t128-dual-t4  koz3-li-tin-59-t128-dual-t4  AA AA0000004  2026-1
```
