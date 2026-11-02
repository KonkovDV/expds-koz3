# PROMPT sat2 — 2nd satellite detector (t225/t228, A → D/W1) 01.11 ~11:45 MSK
Hold **0.41238**. No LB submit. No YOLO / DINOv3 / EN private code. Ideas only. Full GO GPU when A leases.

## Decisions (locked)
1. **Detector:** CenterNet via in-repo `centerhead/` (3-ch sat). **Not FCOS** (extra dep). Backbone: **ConvNeXt-T or R50 via timm** (must differ from current R18 U-Net `spor_*`). ImageNet pretrained OK; no COCO detector weights required.
2. **Classes:** kurgans only (`kurgany_povrezhdennye`, `kurgany_tselye`) for fold0. gor/arch later if pilot PASS.
3. **Tiles:** **grid** cache (`expds.tiles --grid-overlap 0.25`) — not object-centred spor cache (bias). Reuse sat_raw + gt_train_v2.
4. **Fusion:** new group `sp_det2` OFF in default.toml; enable only in side pack. Prefer **extend `sp_score_fusion` to N groups, mean conf in 8 m** with existing sp_or/sp_focal (and sp_focal2 where active). Fallback kill: 12 m add-only if mean fusion hurts precision. Nested thr sweep k10 after OOF.
5. **Gate:** (a) fold0 pilot STOP if vs spor_focal baseline on held-out family: sat kurgan R not ↑ by ≥0.05 **or** P drops >0.03 **or** damaged F1 worse; (b) if PASS → folds 1–4; (c) honest-10 vs B′ 0.315800 triad Δ≥0.003 P≥0.9 damaged≥ before any pack. Cap one GPU until fold0 PASS.
6. **Data:** fold0 = local sat_raw + new grid cache on box/DS `…/koz3-sat2-f0-data`. Full 5-fold = **new Kaggle DS upload** (no S3 secrets until Elena rotates). A acquires lease before DS push/kernel.

## D CPU now (no lease)
- Scratch venv: centerhead tests + 3-ch sat smoke on 4 tiles.
- Manifest region→family→fold from sat_raw + fold0_split.
- Side toml skeleton `[models.sp_det2]` + thresholds OFF — **not** default.toml / main.
- Report STATUS when smoke PASS → A leases GPU for fold0 train.

## A will
- lease slot + kernel slug naming `…/koz3-sat2-f0`
- after lidar seed0 RUNNING, start sat2 fold0 on 2nd free GPU

## Beyond EN (A-executor research, 01.11 ~12:05 MSK — additive to locked decisions above; source EN_PLUS_WORLD_SOTA_20261101.md §4–5)
- **Why heterogeneity is the lever:** EN sat P/R .381/.599 → .442/.642 came from 6 *different* members (D-FINE/RT-DETR/heatmap/
  no-DN/balq) + framing; correlated members gave ≤ +0.008 each (EN WEAKNESSES #22). So sp_det2 must differ in backbone **and**
  resolution/head from spor_* (ConvNeXt-T, stride-2 or stride-4 head with P2 detail path — kurgans are 5–40 px).
- **Training recipe (EN balq + Altai Chen 2026 Remote Sens 18(2):185):** focal/QFL heatmap loss (Altai: focal improved all
  detectors, e.g. AR 0.701 → 0.804); **site-first sampling** (p ∝ n_obj^0.5); **quality aug** (downsample-upsample, Gaussian blur,
  JPEG q30–90, noise, haze, sharpen); background windows only *inside* labelled framing (labels are partial far from the site → no
  far negatives).
- **S0 before training (inference only, hours):** spor_focal d4-TTA OOF (EN single-model sat-nested 0.326 → 0.339) → honest-10;
  kill if Δ < +0.001. Per-member TTA only (EN 8-view TTA on the whole stack regressed).
- **Fusion detail:** mean conf over members, **absent member = 0**, radius 8 m, one nested thr per class (consensus reward).
  Position averaging (WBF) not worth it (EN +0.0004). FR1h framing prior stays orthogonal (applied after fusion).
- **Member #3 (only after sp_det2 PASS):** D-FINE no-DN from public COCO checkpoint trained **by us on Kaggle** (EN: D-FINE fails
  in local env but trains on Kaggle → plausible cause of our sp_dfine "unrepro"); licence check first. FCOS stays out (EN fusion
  0.2729 → 0.2611).
- **Extra gate:** leave-largest-site-out k* ≥ 5 in addition to the triad.
