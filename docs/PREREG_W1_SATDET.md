# PREREG W1-SATDET (02.11.2026, bot-W1-satrec; committed BEFORE any SATDET measurement or GPU run)

Goal: raise satellite recall (sat F1 ≈ 0.25 vs EN ≈ 0.64; biggest hole arkh / damaged kurgans / finds on optics that
never get a candidate). Two tracks, ONE family.

## Context, base, gate (unchanged from SATREC2)
- v3_on b447649, production replay PH (s2b-style replay over a read-only copy of b447649, configs/v3_on.toml),
  honest-10 nested over T["folds"] (orch/bot-W1/work/v3prod/t_taus.json). Base PH 0.345439 ± 0.0005; gate-side base
  V3P 0.3337867 ± 1e-6 (asserted).
- **One family, m = 4 variants (P1 P2 D1 D2). Bonferroni: P ≥ 1 − 0.1/4 = 0.975** (paired family bootstrap p_gt0,
  2000, seed 0, D.site_family).
- PASS ⇔ Δall ≥ 0.003 ∧ P ≥ 0.975 ∧ Δnew34 ≥ 0 ∧ Δnolidar ≥ 0 ∧ P4 per-class F1 floors (kp/kt/gor −0.002, others −0.005).
  Δ, new34, nolidar, floors are measured on the production replay output; P from the gate-side paired bootstrap.
- If a track cannot run (e.g. GPU quota/data blocker), its variants are reported NOT RUN; m stays 4 (conservative).

## Track 2 (CPU): scene-centre prior on an a-priori class list
Lever = `[post].sat_center_prior` (expds/sat_recall.py, flag default OFF) on group sp_dfine before thresholds:
m_c(d) = max(0, 1 − s_c · clip((d − d0)/(1 − d0), 0, 1)), d = scene_drel (sp_or frame boxes; no box ⇒ m = 1).
- **Class list L = {gorodishcha, arkhitektury}, fixed now.** s_c = 0 (no prior) for every other class, kt included.
- Honest disclosure: L was chosen AFTER seeing the published SATREC v1/v2 per-class results (docs/RESULTS_W1_SATREC2.md):
  gor and arkh gained in all 6 SATREC2 variants (gor +0.0091…+0.0671, arkh +0.0145…+0.0530); kp lost in 4/6 and kt failed its
  floor in A1/A2/A3 (kt is excluded by the brief anyway). Reasoning: gor/arkh are large site-scale objects that frame the
  scene, so the imagery was cut around them and edge detections are mostly false; kurgans are numerous and spread to
  the frame edge. fortifikatsii excluded (SATREC2 gains only +0.0003…+0.0022, i.e. noise-level; 2/37 sat coverage); finds/object_poly are not D-FINE
  classes. By per-class independence of thresholds, kp/kt/fort curves are bit-identical to base.
- Grid: d0 ∈ {0, 0.25, 0.5}; s ∈ {0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5}; sp_dfine keeps per-class honest τ (refit nested).
- **P1**: one shared s for L (21 points incl. identity). Nested selection per held fold k: max gate score on the 9
  training folds; ties → smaller s, then larger d0.
- **P2**: independent s_gor, s_arkh (shared d0), evaluated exactly via per-class separability; ties → smaller s_gor,
  smaller s_arkh, larger d0.

## Track 1 (GPU): one more satellite detector, candidates enter fusion BEFORE thresholds
- Architecture (≠ r18 heatmap nets, ≠ sp_focal/sp_focal2, ≠ D-FINE): **two-stage anchor-based Faster R-CNN with
  ResNet-50-FPN v2 backbone** (torchvision `fasterrcnn_resnet50_fpn_v2`, COCO weights, BSD-3; written from the
  torchvision API docs, no EN code, nothing from bots/bot3-*). Anchors 16–256 px (one size per FPN level, ratios
  0.5/1/2), images 640 px fixed (no resize), 5 classes (kt, kp, gor, fort, arkh) + background.
- Data: private acc1 datasets only (elenakonkova/koz3-dino-gor-raw: 175 native SpOr rasters of the 132 dreg regions;
  elenakonkova/koz3-gt-train-v2: GT, markup sp_or polygons only). Rasters warped to UTM 0.5 m. No S3, no credentials.
- **Folds (honest, matching T["folds"])**: 4 models; model j trained on every SpOr region outside held group G_j, where
  G_0 = {f0}, G_1 = {f2, f4, f9}, G_2 = {f3, f5}, G_3 = {f1, f6, f7, f8} (unions of current family folds, balanced by dreg
  count 34/32/31/35). Each region is predicted only by the model of the group containing its T-fold, i.e. by a model
  that saw neither the region nor any region of its family fold. Thresholds: per-class τ of the new group refit by
  D.honest per held fold on the other 9 folds (same as every other group; the training-fold OOF comes from models that
  may have seen fold k — standard OOF stacking, same as sp_dfine).
- Training: SGD momentum 0.9, lr 0.02 (bs 8), wd 1e-4, warmup 300 it, cosine; fp16; time budget ≤ 2 h per
  model (2×T4, 2 models in parallel, 2 waves); 60 % object-centred chips (class weight 1/√n, cap 4) + 40 % random;
  aug: flips, rot90, per-channel colour/gamma, gray, blur, quality aug (downscale 1.5–3.2× + JPEG 15–60 + noise).
- Inference: full rasters, windows 640 / overlap 160, hflip TTA, class-wise NMS 0.5, score ≥ 0.05; box → inscribed
  32-gon ellipse in EPSG:3857; then the same post as sp_dfine in the gate (cluster_circles 4/4, merge_within_markup,
  finalize), threshold_group **sp_det2**, markup sp_or. Extras added to the finish input before thresholds.
- **D1**: sp_det2 all 5 classes, per-class honest τ.
- **D2**: sp_det2 restricted a priori to {gorodishcha, fortifikatsii, arkhitektury} (non-kurgan; kurgans are covered by
  lidar/D-FINE and kurgan floors/new34 failed in SATREC), per-class honest τ.
- Quota: acc1 GPU had 11.39 h left at 02.11 10:05 MSK (kaggle quota); this run is capped at 6 h (deadline in kernel).
  Weights stay in /kaggle/working only (not in git).

## Production placement (flags default OFF)
- Track 2: `[post].sat_center_prior` with `s_by_class` (s = 0 for classes outside L).
- Track 1: new candidate source group sp_det2 (markup sp_or) + per-class τ in `thresholds_by_markup.sp_det2`; flag off.

## Reporting
Per variant: Δ (all / new34 / nolidar), P raw, Bonferroni verdict, class floors, chosen hyperparameters per fold;
detector design, quota used, commits, blockers → orch/bot-W1/STATUS.md.
