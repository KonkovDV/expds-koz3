# PREP 2nd sat detector (FCOS/CenterNet) — bot-D/W1, 01.11 11:43 MSK, CPU-only, no lease/GPU/LB. Hold 0.41238.
Code ref = wt/sp-large-f0 (b27255e, newest expds). Forbidden-stack grep (ultralytics/yolo/dinov3/sam/segformer/mit_/satlas) on all files below: CLEAN.
## Inventory (path — purpose)
- expds/centers.py + expds/train.py + scripts/nn_oof.py — current sat det = R18 U-Net center map (focal, --target centers, r 4 m, min-dist 12 m); nn_oof = fold train+OOF by families (--folds 5).
- centerhead/ (head.py UnetCenterNet/CenterHead hm/wh/off, losses, targets, decode nms_distance, export→feats; 51 tests) — the ONLY in-repo CenterNet; written for lidar (2 kurgan ch, stride 4), channel-agnostic (C=3 ok), smp/timm encoder.
- FCOS / RetinaNet: NOT in repo (only torchvision would provide; not installed on box). Box-detector path that exists: scripts/coco_from_cache.py (cache→COCO/RF-DETR layout, folds by family) → scripts/det_train_hf.py (D-FINE/RT-DETR) → scripts/coco_preds_to_stack.py; expds/det_boxes.py (mask→box, box→circle).
- expds/sp_dfine.py — prior art for a 2nd SpOr detector phase (own group sp_dfine, require_thresholds, cluster_circles 4 m, agreement_m, drop_groups/R); OFF in v4.
- kaggle/2026-10-31/sp_large_f0/{koz3-sp-large-f0.py,eval_fold0.py} + out/sp_large-f0/ (fold0 split, baseline_spfocal_fold0.geojson, control OOF, ANALYSIS_STOP.md) — reusable fold0 harness/gate.
## Merge today (default.toml / pipeline / watchdog)
- [models].sp_or: spor_gor_r18 (gor), spor_focal_r18 (kurgans), surf_arch_r18 (arch); thr [thresholds_by_markup.sp_or] povr 0.3645 / tselye 0.3725. sp_focal2 (256 m window, only no-LAS) thr 0.5016/0.6455, drop_near_kurgans 12 m vs KEPT anchors (pipeline._wide_without_lidar).
- [verify] sp_crop.pt on groups sp_focal/sp_focal2; post.apply_thresholds per threshold_group; post.merge_within_markup (key region,markup,class; isolate only dfine_ae/or).
- post.sp_score_fusion (OFF): mean(conf, max other-group conf in 8 m), HARD-CODED exactly 2 groups (sp_focal, sp_focal2). Plug-in for det #2: new threshold_group (e.g. sp_det2) phase like _wide_without_lidar/_sp_dfine_phase → either 12 m drop vs kept or extend sp_score_fusion to N groups; add group to [verify]/frame_prior lists; own [thresholds_by_markup.<grp>]; default OFF.
## Data (CPU-visible)
- Local: w5data/cache_spor_r1_t128_s25 (201 MB, 3199 tiles/1780 pos, 1 m, t128, 2 kurgan cls — spor_focal training cache, object-centred!), w5data/sat_raw (5.2 GB, 175 SpOr tif/132 regions, pulled from S3), w5data/gt_train_v2.geojson.
- Kaggle DS: elenakonkova|dmitrijkv/koz3-sp-large-f0-data (code+cache+root_fold0 2.1 GB, no creds). Full 5-fold OOF needs all rasters → new ~5 GB DS or S3 path(b) secrets (S3 rotate pending Elena). Full spor_focal_oof.geojson only on F:\expds3\work (not on box).
## Open questions for A PROMPT
1 FCOS (torchvision, new code) or CenterNet (centerhead reuse)? backbone (R18 forbidden as "same family"? R50/ConvNeXt-T/EffNet via timm?) + pretrained weights allowed (ImageNet only? COCO?)
2 classes: kurgans only (povr/tselye) or +gor/arch? 3 train tiles: reuse object-centred cache (coco_from_cache warns bias) or build grid cache (expds.tiles --grid-overlap 0.25)?
4 fusion: 8 m mean-score (extend sp_score_fusion to 3 groups) vs 12 m add-only; thresholds nested k10? 5 gate = fold0 stop (eval_fold0.py) then honest-10 vs 0.31462? 6 data route full 5 folds: DS upload vs S3 secrets.
## Next CPU steps (no lease)
a) install CPU torch/torchvision/smp in a scratch venv; run centerhead tests + 3-ch sat smoke on 4 cache tiles. b) dataset manifest (region→family→fold, tif paths, GT counts) from sat_raw + fold0_split.json. c) config skeleton [models.sp_det2]/[thresholds_by_markup.sp_det2] OFF in a side toml (not default.toml). Needs lease: any training/OOF kernel, DS upload/new S3 use.
## Manifest (01.11 11:54 MSK; PROMPT locked → CenterNet/centerhead, GRID tiles) — sat2_dataset_manifest.json (build_sat2_manifest.py + sat2_grid_augment.py)
- 210 regions in sat_raw, 132 with SpOr tif (175 tif); sp_or GT: povr 1087 (≥1k m² 296), tselye 207 (≥1k 61), gor 65, arch 210, fort 37; every GT region has a local tif.
- Folds = spor_focal recipe (family_folds k=5 on cache_spor positives): f0 «Курганы» 34 reg / 663 kurgans (= sp_large fold0_split ✔), f1 4/135, f2 81/151, f3 25/176, f4 13/169.
- GRID @0.5 m, 512 px (256 m), overlap 0.25: ~5094 windows, ~815 positive (f0 1294/327); full cache ≈4.0 GB uint8 → fold0 ≈1 GB. Native res 0.06–2.12 m → resample needed.
- ⚠ nn_oof recomputes folds from cache positives: with grid positives 15 families move fold → PIN the spor_focal split explicitly (needed for f0 vs baseline gate).
