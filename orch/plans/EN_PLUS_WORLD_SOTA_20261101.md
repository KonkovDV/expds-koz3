# EN_PLUS_WORLD_SOTA_20261101 — EN's ideas + world practice → our lidar retrain and SAT2, ranked by Δ on OUR metric

Author: A-executor (research subagent), 01.11 ~12:00 MSK. Tasks t225u + t227u (Elena FULL GO: «сделай как у них и даже лучше,
изучить мировые практики, реальные кейсы»). Companion: `EN_COMMIT_GAP_VS_US_20261101.md` (t228u, commit-by-commit gap),
`COMPETITORS_GT05_20261101.md`, `out/competitors/adapt/REPORT.md`.
Rules kept: **ideas only** — no EN code/weights (EN README: "private continuation… do not redistribute"); no YOLO/ultralytics,
no DINOv3/SAM3/SegFormer mit_*; DINOv2-L only if later allowed. LB 0.41238 untouched, bot never submits.
Metric reminder: kurgans 50 % (damaged 30 + intact 20), hillforts 15, arch 10, objects 10, finds 10, forts 5; FP ×0.5;
point-in-polygon + coverage. Gap model: LiDAR → EN level +0.053 LB, satellite → EN level +0.051 LB.

## 1. What EN did that moved the score (short; detail in EN_COMMIT_GAP §1)

v5 +0.146 (LiDAR actually runs) · v9 0.417 (sat DETR-family fusion, LiDAR U-Net 2 seeds, masks for hillforts/forts, dual
emission, LiDAR→sat twins) · v11 +0.023 (LiDAR **augx**: height ×0.5–2.2 / zoom ×0.7–1.45 *recomputed from DTM*, site
balance, copy-paste, openness±/SVF, 4-model recipe blend) · **v12 +0.048** (6 heterogeneous sat members, framing prior,
hedge 0.4, LiDAR d8) · v14 (LiDAR neighbourhood prior: R +0.039) · v19b +0.010 (EtGr layers) · v20 +0.003 (Ae forts) · v22 0.50491.
EN's own verdict after v22: post-processing saturated; "future work must add independent spatial or physical evidence rather
than further optimise the same correlated detector scores" (WISE_MLP_TTA_VERDICT).

## 2. World practice (2019–2026), real sources, and what transfers

| # | Source (real, opened/searched 01.11) | Finding (as reported by authors) | Transfer to us |
|---|---|---|---|
| W1 | Berganzo-Besga et al. 2021, *Remote Sens.* 13(20):4181 "Hybrid MSRM-Based Deep Learning and Multitemporal Sentinel 2-Based ML… ~10k tumuli, NW Iberia" https://www.mdpi.com/2072-4292/13/20/4181 | **MSRM (multi-scale relief model) beat slope and SLRM** (AP 63.0 %); Sentinel-2 RF soil mask removed FPs → P 0.97, R 0.64 | add an **MSRM / multi-scale LRM channel** (B3); use a *cheap physical/landcover gate* only as rescoring, never hard-drop |
| W2 | Guyot, Lennon, Hubert-Moy 2021, *J. Archaeol. Sci.: Reports* "Objective comparison of relief visualization techniques with deep CNN for archaeology" https://doi.org/10.1016/j.jasrep.2021.103027 | 13 visualisations benchmarked by CNN; **e2MSTP (multi-scale topographic position + detrending) best** | multi-scale TPI stack as input (B3) — same direction as W1 |
| W3 | Guyot et al. 2021, *JCAA* "Combined Detection and Segmentation of Archaeological Structures from LiDAR…" https://journal.caa-international.org/articles/10.5334/jcaa.64 | Mask R-CNN on eMSTP; models learn "multiscale topographic position" and morphology, not just local maxima | supports heatmap+mask head we already have |
| W4 | Kokalj & Somrak 2019, *Remote Sens.* 11(7):747 "Why Not a Single Image? Combining Visualizations…" https://doi.org/10.3390/rs11070747 ; RVT toolbox https://iaps.zrc-sazu.si/en/rvt | VAT = hillshade + slope + **positive openness + SVF** blend; RVT implements SLRM, SVF, openness, local dominance, MSRM, MSTP | channel set for B3: SVF, pos/neg openness, local dominance (cheap, implementable in our Rust/numpy — no RVT code needed) |
| W5 | ECML PKDD 2021 "Discover the Mysteries of the Maya" challenge proceedings https://www.researchgate.net/publication/362544416 | ALS given as SVF + positive openness + slope; **top-3 = ensembles of per-class models** (DeepLabV3 / UNeXt50 / HRNet) | per-class specialists + ensemble; LiDAR-only beat Sentinel fusion there |
| W6 | Verschoof-van der Vaart et al. 2020, *ISPRS IJGI* 9(5):293 "WODAN2.0: Deep Learning + Location-Based Ranking" https://www.mdpi.com/2220-9964/9/5/293 | **bagging + location-based ranking** improved F1 by 17–35 %; random large test set much harder (barrows ~50 %) than curated (~70 %) | (a) bagging = recipe/seed ensemble (B5); (b) LBR ≈ our framing / neighbourhood priors (B6, FR1h); (c) evaluate on *random* geography = our honest-10/new34 |
| W7 | Verschoof-van der Vaart & Lambers 2019, *JCAA* "Learning to Look at LiDAR: R-CNN…" https://journal.caa-international.org/articles/10.5334/jcaa.32 | Faster R-CNN on SLRM, barrows F1 ≈ 0.67 — single-run numbers on small test areas | never accept a single-seed result; seed variance is part of the gate (EN #20 same) |
| W8 | Chen F. et al. 2021, *ISPRS J. P&RS* 177:217 "Automatic detection of burial mounds (kurgans) in the Altai Mountains" https://doi.org/10.1016/j.isprsjprs.2021.05.010 ; Chen F. et al. 2026, *Remote Sens.* 18(2):185 https://www.mdpi.com/2072-4292/18/2/185 | GF-2 0.8 m; **focal loss vs CE improved every detector** (e.g. DDQ Swin-L AR 0.701 → 0.804, AP 0.523 → 0.681); bigger backbones help; adding typical background samples raises precision | SAT2: focal/QFL loss mandatory, **larger backbone** (ConvNeXt-S/B vs our R18), explicit background windows (hard negatives) |
| W9 | Kazakhstan Jetisu 2026 (Arch. Kazakhstan, doi 10.52967/akz2026.1.31.290.310) https://doi.org/10.52967/akz2026.1.31.290.310 | Sentinel-2 L2A monthly composites, 6 bands, U-Net; independent-year test P 0.49 / R 0.86 | our SpOr is RGB(+alpha) 0.1–3 m (EN: "NDVI/NDWI/SWIR impossible") → **not transferable**; multispectral only as future external layer (licence check) |
| W10 | *J. Documentation* 80(5):1167 (2024), Bulgaria burial-mound CNN field validation https://www.emerald.com/jd/article/80/5/1167/1236195 | IKONOS CNN tile classifier: field validation FN 95–96 %; **training only on highly visible mounds performed worse** | keep *all* labelled mounds incl. faint/damaged in training; don't filter "clean" positives (consistent with EN partial-label failures) |
| W11 | Ghiasi et al. 2021, CVPR "Simple Copy-Paste is a Strong Data Augmentation Method" https://arxiv.org/abs/2012.07177 | copy-paste strong for rare instances | already in cnx (paste_bank); EN attenuated paste (30–100 % height) was **rejected** → keep full-height paste |
| W12 | Solovyev et al. "Weighted Boxes Fusion" https://arxiv.org/abs/1910.13302 ; Esri ArcGIS GeoAI blog on test-time augmentation | fusing heterogeneous models' outputs > NMS; TTA helps under shift | SAT2 fusion = mean confidence over members with absent = 0 (EN: WBF positions only +0.0004 → don't bother with position averaging) |
| W13 | Lin et al. 2017 Focal loss https://arxiv.org/abs/1708.02002 ; Shrivastava et al. 2016 OHEM https://arxiv.org/abs/1604.03540 | hard-example emphasis | B4 hard-negative windows (LiDAR labels cover full tiles → safe; satellite labels only near site → **unsafe**, see framing) |
| W14 | Sagawa et al. 2020 Group DRO https://arxiv.org/abs/1911.08731 | worst-group robustness via group re-weighting | site = group; B2 temperature-balanced sampling is the cheap version; DRO-reweight only as optional pilot |
| W15 | Hinton et al. 2015 Knowledge distillation https://arxiv.org/abs/1503.02531 | ensemble → single student | **runtime lever**: if the 3-recipe LiDAR blend busts the 600 s/region budget, distil to one ConvNeXt-T student (B7) |
| W16 | arXiv 2307.03512 "Transfer Learning of Semantic Segmentation… Buried Archaeological Structures on LiDAR" https://arxiv.org/html/2307.03512v4 | ImageNet init ≥ scratch; archaeology-to-archaeology transfer inconsistent | test `in` (ImageNet) init vs our `sc` scratch on the new recipe (cheap: one fold) |
| W17 | URFU 2026 "Semi-automatic artificial kurgans detection with YOLO and foundation models" https://doi.org/10.1109/usbereit70063.2026.11580650 | RU kurgans, YOLO + foundation models, semi-automatic | YOLO banned for us; only idea = verification stage (we have crop verify) |

Not transferable / already closed: raw LAS returns (EN AUC 0.41), persistent homology, morphometric GBM rescoring, per-site heat
recalibration (EN: all variants below deployed), test-time self-training (0.564 → 0.498), partial-label pseudo-labels, WiSE-FT,
zoom-TTA on satellite (−0.06), Google Earth augmentation (ToS), Sentinel-2 multispectral (we have RGB only).

## 3. Ranked adoption list for OUR metric (expected LB Δ = hypothesis, not measured; ranges from EN ledger × our gap model)

| Rank | Upgrade | Basis | Exp. LB Δ | Stack fit | GPU |
|---|---|---|---|---|---|
| 1 | **SAT2: heterogeneous 2nd (and 3rd) satellite detector + mean-score fusion** | EN v12 (+0.048 incl. framing), Altai W8 (focal + bigger backbone), Maya W5 | +0.02…+0.05 | new group `sp_k2` fused with sp_focal at 8 m before thresholds | 1–2 GPU-days |
| 2 | **LiDAR R-aug: DTM-level height/zoom + relief recompute + site balance** | EN LI_AUG (+0.016 li-nested), WEAKNESSES #21 | +0.01…+0.025 | cnx trainer, flags default-off | 5 folds × 12k ≈ 8 h T4×2 |
| 3 | **LiDAR R-augx channels: neg-openness, SVF, MSRM/multi-scale LRM** | EN augx (+0.020 single), W1/W2/W4 | +0.005…+0.015 | 7-ch variant; c5 tags + 2 new | same |
| 4 | **Recipe-diverse LiDAR blend (R-aug, R-augx, R-ms) + neighbourhood prior in decoder** | EN blend +0.008, prior +0.0066 li-nested / R +0.039 platform; W6 bagging | +0.005…+0.015 | average logits before decode; prior = existing `kurgan_neighbour_prior` | inference only |
| 5 | Satellite d4-TTA on spor_focal (OOF) | EN 7c70d5f 0.326 → 0.339 | +0.003…+0.01 | `sp_or` inference flag | hours |
| 6 | LiDAR hard-negative windows (round 2 mining) | li_r18_hard2 precedent (deployed), W13 | +0.002…+0.008 | sampler bank of top FP peaks from inner-fold OOF | +1 run |
| 7 | ImageNet init vs scratch for cnx | W16 | ±0.005 | flag `init=in` | 1 fold |
| 8 | Distillation of blend → 1 student (only if budget fails) | W15 | 0 (keeps Δ of #4 inside 600 s) | — | 1 run |
| 9 | Hillfort localiser DINOv2-L (if allowed) | EN v23/24 | +0.005…+0.02 | T8 | later |

## 4. Upgrades BEYOND EN (what EN did not do, or did and we can do better)

LiDAR (detail + flags in `PROMPT_LIDAR_AUG_SEEDS_20261101.md`):
- **L+1 Discrete DTM-variant relief banks** instead of per-crop recompute: precompute relief for each training site at
  (zoom, height) ∈ {0.75, 1.0, 1.3} × {0.6, 1.0, 1.7} on the *residual* DTM (z − lowpass₅₀ₘ)·h + lowpass, then sample a bank per crop
  + small continuous jitter (±10 %). Same physics as EN, ~0 dataloader cost, deterministic parity with inference (h = z = 1 bank
  is exactly the inference relief). EN recomputed per crop (slow on CPU-bound loaders).
- **L+2 Multi-scale relief channel (MSRM / multi-scale LRM 5–40 m)** — EN never tried; W1/W2 show it is the best single
  visualisation for mounds of varying size. Targets EN's failure mode (Novenkoe/Erentuna large/tall mounds "resemble nothing else").
- **L+3 Temperature site balancing** p(site) ∝ n_obj^0.5 (α = 0.5) rather than pure uniform: keeps dense cemeteries informative
  while lifting rare sites; option α = 0 (uniform, EN-like) as an ablation on f0 only.
- **L+4 Recipe diversity over seed count**: EN's data say a 2nd augx seed gave nothing but different recipes gave +0.008; we spend
  3 GPU slots on 3 recipes (R-aug, R-augx, R-ms) × seed 0, not 3 seeds.
- **L+5 Fold-dispersion of the selected threshold as a kill metric** (our cnx failed on per-family thr 0.329 vs 0.064).
- **L+6 Replacement-first deployment** (b-kur): the new blend replaces li-kurgan groups; additive is secondary.
- **L+7 Distil-if-needed** to hold runtime (W15).

SAT2 (stack fit: our sp_or tiles, folds, `nn_oof`-style OOF export; no YOLO/DINOv3):
- **S+1 Different family, not another R18 focal**: EN's correlated members gave ≤ +0.008 each (WEAKNESSES #22), heterogeneity
  gave the +0.048. Candidates in order: (a) **ConvNeXt-S/B U-Net centre-heatmap with QFL at stride 2 (P2 detail path)**, timm
  weights (Apache/MIT-compatible), site-balanced + quality-aug — closest to our code, different backbone/resolution;
  (b) **D-FINE-L no-denoising from the public HF COCO checkpoint, trained by us on Kaggle** (EN: D-FINE trains fine on Kaggle but
  silently fails locally — likely our "unrepro" sp_dfine cause; licence Apache-2.0 — confirm allowed; we do NOT use EN/previous-stage
  weights); (c) torchvision FCOS — **lowest**: EN measured FCOS lowering fusion (0.2729 → 0.2611).
- **S+2 Quality augmentation** (resolution loss/blur, JPEG, noise, haze, sharpen) + site-first sampling — EN balq +0.0075 sat-nested.
- **S+3 Focal/QFL + explicit background windows *inside* labelled framing** (W8); never mine negatives far from the labelled zone
  (labels exist only near the site — EN #23, our FR1h).
- **S+4 Fusion = mean confidence over members, absent = 0, radius 8 m, one threshold per class chosen nested** (EN v12) — this
  is a consensus reward (Family «≥2 of 4»); orthogonal to FR1h framing prior.
- **S+5 d4-TTA per member** on OOF only if inference budget allows (EN gain on single model; 8-view on a full stack regressed → test per member, not stack).

## 5. SAT2 pilot spec (no PROMPT_SAT2 existed at 12:00 — this section is the prompt source)

- S0 (hours, inference only): spor_focal d4-TTA OOF on the existing folds → honest-10 gate. Kill if Δ < +0.001.
- S1 member A = ConvNeXt-S U-Net QFL heatmap stride 2 (kurgany_povr/tselye), site-balanced, quality-aug, 4 folds = sp_or folds,
  OOF export in our geojson format (threshold_group `sp_k2`). Early stop: fold-0 kurgan F1 alone ≥ 0.8 × spor_focal fold-0 F1.
- S2 fusion study (CPU): members {sp_focal, sp_focal2, sp_k2} mean-score, absent=0, r 8 m; thr nested per class; compare vs
  current stack on honest-10 + new34 + no-lidar.
- S3 (only if S2 ≥ +0.003): member B = D-FINE no-DN (if allowed) → 4-member fusion.

## 6. Kill gates and pilot protocol (both lines)

1. Pre-register in `orch/bot-W2/PREREG_W2_cnx.md` (LiDAR) / new `PREREG_SAT2` before any measurement.
2. Early stop on f0+f1 li-only (LiDAR) / fold-0 (SAT2) vs the matching baseline (c3-sc s2: 0.2198 / 0.4786; li-only mean ≤ 0.3492 → stop).
3. Full gate = existing protocol (honest-10 by family, paired bootstrap 2000 / 103 families, `materials/compute_metrics.py`, base A):
   Δ ≥ +0.003 **and** P ≥ 0.9; class floors −0.002 (povr/tselye/gor), −0.005 others; Δnew34 ≥ −0.001; Δno-lidar ≥ −0.001.
4. **Added (EN robust acceptance):** leave-largest-contribution-out k* ≥ 5 sites; fold dispersion of the
   chosen threshold ≤ ×2 (L+5); blind-site recall reported (sites with li recall < 0.2 at baseline).
5. Seed variance: a single-seed PASS is "candidate"; promotion needs the recipe blend to pass too.
6. Runtime: full-system rehearsal ≤ 600 s per region budget (A1 rules) before packaging. Packaging / LB only with Elena «да».
