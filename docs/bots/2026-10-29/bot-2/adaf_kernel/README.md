# ADAF fine-tune kernel (bot-2, task 10) — NOT submitted

Fine-tunes the ADAF Faster R-CNN barrow detector (Irish ALS) on КОЗ-3 lidar SLRM tiles for
`kurgany_tselye`, `kurgany_povrezhdennye`, `fortifikatsii` (optionally `gorodishcha` via `ADAF_CLASSES`),
out-of-fold by family (bot-9 `region_fold.json`, folds {0:6, 1:1, 2:13, 3:18, 4:21} regions), and exports
bot-9-compatible maps + OOF vectors.

Status: code + CPU smoke test only. Nothing pushed, nothing submitted to Kaggle, no LB.
The zero-shot result that decides whether this is worth a GPU run is in `../REPORT2.md` (section «Задача 10»).

## Files
- `adaf_finetune.py`: single-file kernel (torch + torchvision ≥0.15, numpy, shapely ≥2, pyproj, scipy). Own code.
- `region_fold.json`: copy of `bot-9/kaggle/convnext_lidar/c3sc_f01_out/region_fold.json` (59 regions, family folds).
- `smoke_cpu.sh`: CPU smoke test (1 thread) on 2 tiny regions, 2 iterations, 256 px crops.
- `check_maps.py`: checks maps npz keys/shapes/dtypes against bot-9 HEAD_INTERFACE.md and counts the decoded peaks.

## Inputs (Kaggle datasets to create; no S3, no credentials needed)
1. `koz3-adaf-slrm`: `bot-2/adaf/slrm/*.npz` (59 files, ~0.1 GB), built by `repo/scripts/adaf/prep_slrm.py`.
   The recipe: ground (class 2) mean z on a 0.5 m grid, interior holes filled, SLRM r=20 px, clip [-0.5,0.5] → u8.
   This matches ADAF's input (SLRM 0.5 m, [0,1], repeated to 3 channels).
2. `koz3-gt`: `gt_train_v2.geojson` (EPSG:3857).
3. ADAF init: the kernel downloads `OD_barrow.tar` from Zenodo (internet on), with md5 b9afba22…ab80 checked.
   You can also attach it as a dataset and set `ADAF_INIT`.

## Recipe (defaults; env-overridable)
- Model: torchvision `fasterrcnn_resnet50_fpn_v2`, 1+K classes, min_size 800.
  It loads everything from ADAF OD_barrow except the box predictor. Every new class row is initialised from
  the ADAF "barrow" row (plus 1e-3 noise), and the bg row from bg.
- Training per fold k: regions with fold ≠ k. 512 px crops: 70 % centred near a GT object (±1/3 crop jitter),
  30 % uniform with ≥30 % valid. D8 augmentation. GT boxes come from the polygon bounds (min 2 m), and only
  objects whose representative point is on valid lidar are used.
  SGD lr 0.01, momentum 0.9, wd 1e-4, warmup 300, cosine, 4000 iterations × batch 4, AMP on GPU, grad clip 10.
- Inference on fold-k test regions: 512 tiles, step 384, score ≥0.05, class-wise NMS IoU 0.3, then the
  box centre must be on valid lidar.
- Rough cost: T4 GPU ~4–6 it/s at batch 4 → about 15–20 min per fold, plus inference. This is an estimate, not measured.

## Outputs (`/kaggle/working/out`)
- `maps/maps_adaf_{tag}_{region}.npz`: written only for the region's test fold. Same keys as bot-9 maps:
  `valid` u8 (H,W), `valid_h` u8 (H/4,W/4), `hm` f32 (2,H/4,W/4) with 0=tselye and 1=povr,
  `wh` f16 (1,·) = log(radius in head px), `off` f16 (2,·), `mask` u8 (3,H,W) with 0=gor, 1=fort, 2=arch.
  `meta` is a JSON string with region, fold, family, tag, head="adaf_frcnn", stride 4, left, top, step 0.5,
  step_head 2.0, crs, H, W, classes.
  Each detection is written as a single peak (value = score) in its centre cell, so a pool-3 peak decoder
  (centre = left + (col+0.5+off_x)·step_head) gives back the box centres.
  fort/gor detections are painted as score·255 inside the box in `mask` (and in `arch`).
- `oof_adaf_{tag}.geojson`: EPSG:3857 1 m circles, with props class_name, confidence, region_name, crs,
  markup_type "adaf", threshold_group "adaf_ft", original_crs, fold, radius_m. fid starts at 30,000,000.
  The format is the same as the zero-shot splice (`repo/scripts/adaf/adaf_splice.py`), so it splices into the stack as
  its own threshold group.
- `region_fold.json` (folds + jobs), `report.json` (loss curve, per-fold and OOF raw recall/AP per class),
  `weights/adaf_ft_f{k}.pt` + `ATTRIBUTION.txt`.

## Notes for bot 9
- The maps follow HEAD_INTERFACE.md but are detector outputs rendered as sparse peaks, not dense heatmaps.
  Use them for candidate generation and score features, not as a dense-feature map.
- Thresholds: give the new group (`adaf_ft`) its own out-of-fold threshold per (group, class) in the stack.
- No S3 is used, so there are no secrets to wire. If you embed this into your build.py, keep the dataset
  inputs as above.

## Licence and attribution
- ADAF weights: Kostovska, Kocev, Čož, Kokalj, Curran, Corns, "ADAF pretrained models",
  doi:10.5281/zenodo.15848663, **CC-BY-SA-4.0**.
  The fine-tuned weights this kernel produces are adaptations, so they **must be released under CC-BY-SA-4.0**
  with attribution. The kernel writes `weights/ATTRIBUTION.txt` and puts a licence string in each checkpoint.
  Cite: Čož et al. 2026, *J. Archaeol. Sci.: Reports* 71, 105733, doi:10.1016/j.jasrep.2026.105733.
- Code: written by bot-2 from scratch on torchvision. No code is copied from ADAF (Apache-2.0,
  github.com/EarthObservation/adaf @ a44acdd) or from aitlas (Apache-2.0).
  The architecture choice and the input convention (SLRM 0.5 m, [0,1], 3× repeat) follow the ADAF paper and repo.
  If the competition requires open-sourcing, the CC-BY-SA share-alike applies to the weights, not to this code.
