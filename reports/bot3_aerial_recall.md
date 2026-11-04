# Aerial recall on LB v2 (0.013)

Branch `bots/bot3-aerial-recall`. No leaderboard submission. All new flags default off.

## 1. Why recall is 0.013

LB v2 (`eb268d6`, log `docs/background/LB_SCORING_v2_eb268d6_0.41238.log`) prints аэрофото precision 0.090909 and recall 0.013333. The smallest integers that match are **75 ground-truth objects, 1 true positive, 20 false positives, 21 predictions** (`docs/LB_LOG.md`). The same six digits have been printed since the 24 October submission. The coordinator’s e2e check is the same fact on another set: **v3 aerial F1 0.096 equals v2**. v3 adds D-FINE on satellite/ortho (`[sp_dfine]`, default off in the config that was measured with the flag on for SpOr only). It does not add an aerial kurgan head.

### The shipped aerial head only emits fortifications

`configs/default.toml`: `[models] ae` is `models/ae_fort_mask_r18.pt` with `classes = ["fortifikatsii"]` and `[thresholds_by_markup.ae] fortifikatsii = 0.998215`. On the train audit (bot-6, `docs/bots/2026-10-28/bot-6/REPORT.md`) aerial ground truth is **1749 kurgans out of 2179 objects (80%)**, plus 393 forts, 19 hillforts, 18 architecture. The fort mask at that threshold keeps 714 polygons and hits **94 of 696 forts** (`notes/ae_fort_mask_2026-10-22.json`). One hit out of 75 public objects is that operating point, not a crashed network. A previous aerial kurgan head was measured at **+0.00004** and was not kept (`notes/ae_focal_2026-10-21.json`).

### Lidar copies do not fill the gap, and intact mosaic copies are switched off

`expds/fusion.py` `li_to_ae` runs only when the region has LAS (`expds/pipeline.py`). The public set has aerial imagery on **4 regions and LAS on 1** (`docs/LB_LOG.md`). Even a perfect lidar copy reaches **595 of 1749** aerial kurgans on train (34%). Mosaic and coarse copies are tagged `li_ae`. `[thresholds_by_markup.li_ae] kurgany_tselye = 1.000001`, so an intact copy from the mosaic never passes. That matters because a 1940s mound is often still intact while the lidar class is “damaged”, and the metric matches **by class**. A damaged lidar polygon does not count for an intact aerial ground-truth object. The 24 October copy patch did not move the printed aerial precision or recall, so those copies were not in the public file.

`[post] kurgan_twins_only = true` does **not** delete aerial copies: it forces `dedup_ae_vs_li_m = 0` (`expds/watchdog.py`).

### Score / NMS of the fort mask is not an accidental zero

0.998215 is the joint optimum from the five family folds, not a floor left at 1. The U-Net assigns each polygon to one window core; it does not run a class-agnostic NMS that would erase the class. `[ae_recall_relax]` (already in the tree, default off) lowers the fort threshold only to 0.995. That cannot create kurgan candidates.

### CRS shift is real and too small to explain 0.013

Bot-6 measured a stable lidar-to-aerial shift on 4 of 14 regions, median about 8.5 m, and **611 → 634** centres inside an aerial polygon (+23, about 1.7%). `[fusion] register` exists and is off. 2170 of 2179 aerial ground-truth points fall on a valid aerial pixel, so the labels and the frames agree. Two train frames are EPSG:3857 and are read in file units because the shipped weights were trained that way (`pixel_units` is not `ground`). `[raster_fix] reproject_3857` is off; turning it on would change the scale relative to those weights. The geographic-CRS bug (008 Герите, EPSG:4326) does not affect aerial: all 32 aerial regions have a projected UTM.json (`docs/bots/2026-10-29/bot-9/DFINE_FINAL_ORAE.md`).

### Class ids are not swapped

D-FINE `id2label` is `kurgany_tselye`, `kurgany_povrezhdennye`, `gorodishcha`, `fortifikatsii`, `arkhitektury`, the same tuple as `expds/sp_dfine.py` `CLASSES`. The aerial OOF (`docs/bots/2026-10-29/bot-9/dfine_ae_oof/ae_oof_analysis.json`) hits both kurgan classes and forts, so the names are not a permutation. The fort checkpoint is a single class and does hit forts.

### D-FINE Ae is the missing detector, and its default fusion does not match the OOF

The model is not in the v2 submission. The slot is `[dfine_aeor] enabled = false` (`expds/dfine_aeor.py`, present at v4 `cbd4804` and on this tree). OOF at score 0.05: recall **0.47** (938/1994) and about **1.92 million** false positives. Usable points on that file: damaged kurgans F1 about **0.21 at 0.6** (recall 0.17), forts **0.12 at 0.5** (recall 0.12), hillforts none above 0.2, architecture 0/1. Large objects are seen at low score (all 52 above 5000 m² at 0.05, any class) and disappear once the threshold rises. v4 sets `[thresholds_by_markup.dfine_or] gorodishcha = 1.01`. That disables ortho hillforts on purpose; it does not zero aerial kurgans.

Two infer/train mismatches, both default off behind `[dfine_ae_tile]`:

| | OOF kernel `kaggle/2026-10-29/dfine_spor/ae_o2c/koz3-dfine-ae.py` | `dfine_aeor` default |
|---|---|---|
| View fusion | one `batched_nms`, raw score kept | WBF: score sum / max(n_views, members). Flip TTA has 2 views, so a box seen once is halved |
| Window grid | `range(0, max(1, n-overlap), step)` | last window pinned to `n-tile` (width 1000 → start 480 vs 360) |

`tests/test_bot3_aerial_recall.py` checks the halving: score 0.62 stays 0.62 under `fuse=nms` and becomes 0.31 under `fuse=wbf`. Thresholds swept on the OOF would then drop the box. NMS in both paths is per frame. Overlapping frames of one region (train: 101 has two, 103 has three) stay as extra false positives. `[ae_recall] cross_tif_dedup_m` is the opt-in for that.

## 2. Flags

| Flag | Default | When on |
|---|---|---|
| `[ae_recall] mosaic_intact` | `false` | Intact copies that would be `li_ae` are tagged `ae`, so they use `kurgany_tselye = 0.802255` instead of `1.000001`. Damaged copies stay `li_ae`. The fort threshold is not moved. |
| `[ae_recall] cross_tif_dedup_m` | `0` | Among `dfine_ae` / `dfine_or` (and the `ae_dfine` / `or_dfine` aliases), keep the higher score when centres are within this many ground metres, same region and class. `0` returns the same list. |
| `[dfine_ae_tile]` | `false` | Already in the tree. Patches `[dfine_aeor]` to `windows_mode=train` and `fuse=nms` so inference matches the OOF. Does not enable D-FINE by itself. |
| `[dfine_aeor]` | `false` | The aerial/ortho detector. Needs weights and an OOF threshold file. |
| `[ae_recall_relax]` | `false` | Already in the tree. Lowers the fort gate to 0.995 and the lidar-copy gates. Not the lever for the missing kurgan class. |
| `[fusion] register`, `[raster_fix]` | off | Shift and 3857/int16/nodata reads. Measured effect of the shift is about +23 centres. Leave off until honest-10 says otherwise. |

With every flag off, `li_to_ae` groups and `finish()` outputs match the previous code (`tests/test_fusion.py`, `tests/test_dfine_aeor.py::test_flag_on_old_groups_byte_identical_to_off`).

## 3. Honest-10

**Not evaluated. The train stack (`F:\expds3\work` or `/workspace/data`) is not in this environment.** Run:

```text
python -X utf8 scripts/eval_ae_recall.py --work F:\expds3\work --train F:\2\train-expds-3 --expect 0.314615 --honest 10 --dfine-oof F:\expds3\work\oof_dfine_ae.geojson --out F:\expds3\work\ae_recall_honest10.json
```

The script exits 2 and writes nothing if those files are missing. It does not call a GPU and does not submit.

Until that JSON exists, baseline and delta for recall, precision and F1 are **pending**.

## 4. Run request (Dmitrij’s PC, no Kaggle)

- **Branch:** `bots/bot3-aerial-recall`
- **Machine:** the PC that already has `F:\expds3\work` and `F:\2\train-expds-3`. CPU only. Do not start a Kaggle job: the OOF file is the NMS-protocol scores, and the GPU budget quoted through 31 October is already closed. Do not retrain.
- **Command:** the block in section 3. If the OOF lives elsewhere, pass that path. Drop `--dfine-oof` to score only `mosaic_intact`.
- **Runtime:** on the order of one `scripts/scoreboard.py --honest 10` of base A, times the number of variants (2 without the OOF, 4 with it), plus one official metric pass per variant on the thresholded file.
- **Success:** process exit 0; `ae_recall_honest10.json` contains `variants.baseline.honest10` within 0.0005 of **0.314615**; each other variant has `delta.honest10` and, unless `--skip-aerial`, `delta.aerial_recall`, `delta.aerial_precision`, `delta.aerial_f1`. A variant is a candidate only if honest-10 Δ ≥ +0.002 and the family bootstrap `p_gt0` ≥ 0.8, and aerial recall does not fall.
