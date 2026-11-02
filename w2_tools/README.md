# w2_tools — скрипты W2 (копия `/workspace/koz3/w2/tools/`; дубль в `tools/w2/`)
| Скрипт | Что | Пример |
|---|---|---|
| `collect_cnx.sh` | скачать output ядер c3-sc (read-only) | `collect_cnx.sh <acc> <slug> <dir>` |
| `merge_cnx.py` | склеить фолды в один OOF с проверками | → `oof/merged/oof_cnx_*.geojson` |
| `seed_merge_cnx.py` | усреднить сиды | → `*_s012avg.geojson` |
| `blend_maps_cnx.py` | P-W2-12: бленд карт N моделей (hm/mask/wh/off среднее, valid AND, проверка fold/H/W) + один декод | `blend_maps_cnx.py --report <report_cnx_full.json> --member s0=GLOB ... --out x.geojson --group li_cnxblend` |
| `stack_add_cnx.py` | A + OOF как группа li_cnx | `stack_add_cnx.py stack_arch.geojson oof.geojson out.geojson cnx_s2` |
| `stack_replace_li.py` | заменить li-группы на OOF | `… <group> [--classes kurgany_…]` |
| `gate_detail.py` | разбор гейта по классам | `gate_detail.py base_counts.pkl counts_X.pkl out.json` |
| `center_snap.py` | P-W2-4 (FAIL) | — |
| `ci23_ground_cut.py`, `las_cls_probe.py` | аудит классов LAS (CI2-3) | — |
| `ci28_hs_fit.py` | hillshade parity (CI2-8) | — |
| `pull_maps.sh` | скачать maps/ из ядер | — |
| `early_stop_r50c5.py` | ранний стоп P-W2-9 по f0/f1 (правила 1+2, базы в `w2_notes/r50c5_early_stop_baseline.json`) → PASS/FAIL | `$K/venv/bin/python w2_tools/early_stop_r50c5.py <output f01>` |
| `s3get.py` | скачать из S3 train-expds-3 (env `S3_ENDPOINT`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` — значения из приватного датасета koz3-s3-cred, не печатать) | — |
