# COMPETITORS_GT05_20261101: how the competition got above 0.5, and what we take from it

Author: A (executor), 01.11 11:40 MSK. Task t224u (Elena): «Проработай всё максимально глубоко, изучить код репо конкурентов, их коммиты, как там он вышли на балл выше 0.5 и сделай также у нас».
Status: research is done; the CPU diagnostics are done (`out/competitors/adapt/`). **LB 0.41238 held.** No submit, no push, no GPU, no change to default.toml, no messages to Elena.

## 0. Rules, and a conflict that needs Elena's decision

- PLAN_TO_1111 already says: «Чужой код не копировать. У публичных репозиториев лицензии нет… EN — ничего не брать» (ideas only, clean-room). YOLO/ultralytics/DINOv3/SAM3/SegFormer mit_* are banned.
- t224u says «изучить код репо конкурентов… сделай также у нас». I resolved this as follows: **I read EN's docs, commit messages and ledger (README, EXPERIMENTS.md, WEAKNESSES.md), and no `.py` file went into repo/.** The clones are blobless and read-only, outside the repo: `out/competitors/сторонний репозиторий_EXpedition` (heads 90a7460 / codex/v24-dinov3-1024) and `out/competitors/yakirill99_ds-expedition` (346f22a).
- EN's README (branch codex/v24) calls the repository a "private continuation… Do not make public/redistribute". It contains legacy/previous_team code and weights from the previous stage. **Weights and code from there cannot be used in any form.**
- **Question for Elena (not asked; the parent decides how to ask):** does t224u loosen the "EN — ничего не брать" rule? Until she answers, everything below is an idea-level transfer with our own implementation.

## 1. Who is above 0.5 (sources only)

The LB is public on expds-platform.upgreat.one (Leader-ID login). Snapshot as of 29.10, from `repo/docs/AUDIT_CLAUDE_2026-10-29.md §2` (28.10 is in `docs/research/2026-10-28/OSINT_TEAMS.md`):

| # | Team | Score | Public materials |
|---|---|---|---|
| 1 | Nikolove | 0.61636 | none |
| 2 | CVшники | 0.57996 | none |
| 3 | accuracy=0.5 | 0.56855 | none |
| 4 | Family | 0.55848 | 1st place in KOZ-2 «Сканирование». Habr 1054936: 8 models (4 surface + 4 geo), an object is accepted if ≥2 of 4 models agree, Monte-Carlo resampling for model selection (stability > peak), scale/brightness/rotation augmentations |
| 5 | GigaFlex | 0.54973 | 3rd place in KOZ-2. Habr: 6 models, a universal tile model followed by single-class specialists; "work with the data, not the code" |
| 6 | Фанаты Фишмана | 0.53492 | none |
| 7 | MTs | 0.52058 | none |
| 8 | котики | 0.51154 | none |
| 9 | Сахалинский центр ИИ | 0.50491 | **public repo github.com/сторонний репозиторий** (the only above-0.5 team with code and commits) |
| 10 | Землекопы | 0.50232 | none |
| — | (mlart, 2nd in KOZ-2) | — | Habr: models split by physical meaning (relief / LiDAR reflectance / magnetometry), several model families |
| 35 (29.10) | StrataTrace (us) | 0.34796 → now **0.41238** | — |

On 01.11 the web search turned up nothing new (no GitHub solutions, no new posts on t.me/s/expds). `yakirill99/ds-expedition` is the baseline level; its ideas are already covered (COMPETITORS_2026-10-29.md).
Not invented: team names and scores come only from the snapshot above. Habr facts come from https://habr.com/ru/articles/1054936/ (opened 01.11).

## 2. How EN got from 0.30 to 0.50+ (from EXPERIMENTS.md and the commits)

| Step | What changed | LB Δ | Evidence |
|---|---|---|---|
| v5 | laspy reads LAS at all (infrastructure) | **+0.146** | EXPERIMENTS.md |
| v12 | **Satellite: fusion of 4–6 detectors** (D-FINE/RT-DETR family from the previous stage + FCOS). Merge radius 8 m, score = mean over members, threshold 0.25; framing prior; hedge of the second kurgan class; hillfort snap | **+0.048** | docs/bots/2026-10-28/bot-5/NOLIDAR.md (sat-nested 0.2585→0.3797) |
| v14 | LiDAR heatmap/CenterNet ensemble (multi-seed, site balance +0.012–0.016, height ×0.5–2.2 / zoom ×0.7–1.45 with relief recomputed, D4-TTA) | LiDAR R 0.64→0.676 | d54dd1d, WEAKNESSES.md #20–21 |
| — | satellite d4-TTA | sat-nested 0.326→0.339 | 7c70d5f |
| v19b | EtGr layer isolation + pixel-GBM geo objects | +0.010 / +0.003 | 0d84268 (ETGR_LAYER_FIX) |
| v20 | Ae fortifications via FCOS | +0.003 | 609de3b |
| v22 | = **0.50491** | — | snapshot |
| v23/v24 | DINOv3-SAT hillfort localiser (sat-nested 0.3797→0.4097/0.4159) → v24 0.537 (internal ledger) | ~+0.03 | c58d053, 3e12e1b. **DINOv3 is banned for us** |
| 29.10 | rejected: raw LAS returns, WiSE, LSKNet, DINOv3 arch/SAT detector, partial-label pseudo-kurgans, masked loss | 0 | 346978e, 0726c60, 80240aa, 90a7460 |

### 2.1 What actually explains the gap between 0.412 and 0.505 (`out/competitors/adapt/gap_model.py`)

Per-modality public P/R, ours (eb268d6 log) vs EN v22:

| Modality | Ours P / R | EN P / R | LB Δ if we were at EN level |
|---|---|---|---|
| LiDAR (482 GT) | 0.516 / 0.504 | 0.596 / 0.676 | **+0.053** |
| Satellite (377) | 0.456 / 0.255 | 0.434 / 0.639 | **+0.051** |
| Geophysics (1441) | 0.769 / 0.656 | 0.754 / 0.724 | +0.005 |
| Ortho (50) | 0.368 / 0.14 | 0.71 / 0.24 | +0.0045 |
| Ae (75) | 0.09 / 0.013 | 0.10 / 0.09 | ≈0 |
| **LiDAR + satellite** | | | **+0.094 ≈ the whole gap of 0.093** |

The proxy is calibrated: real/proxy is 0.879 for us and 0.871 for EN.
**Takeaway: EN is above 0.5 because of kurgan recall on LiDAR and satellite at the same precision. That is model quality, not post-processing.** EN says the same about itself (WEAKNESSES.md §E: "Post-processing… saturated"; NOLIDAR: "limited by the detectors").

### 2.2 Our CPU diagnostic: thresholds or detector? (`out/competitors/adapt/kurgan_ceiling_by_markup.py`, honest OOF stack A)

| GT markup | GT | Ceiling (any candidate) | R at eb268d6 thresholds | Satellite with relaxed thresholds |
|---|---|---|---|---|
| sp_or (kurgans) | 1294 | **0.65** | 0.213 (P* 0.34) | ×0.5: R 0.317 / P* 0.145; ×0.25: R 0.453 / P* 0.064 |
| li | 2154 | 0.82 | 0.552 (P* 0.54) | — |
| ae | 1749 | 0.19 (own candidates) | 0.204 | — |

The right kurgans are in the candidate pool (the satellite ceiling matches EN's LB recall). Our confidence ranks them poorly, though. Lowering thresholds destroys precision, and an LGBM reranker over the same features is already closed (w5_rerank V5: kurgans −0.0005, P 0.50). **No CPU fix exists for the satellite. It needs new or different detectors with better ranking.**

## 3. Inventory of the >0.5 techniques against our code

| Technique (source) | Us | Status |
|---|---|---|
| Separate expert per data type, threshold per type/group (EN, mlart) | yes, thresholds_by_markup, groups | **have** |
| Framing prior on satellite (EN +0.028 sat-nested) | FR1h `[post.frame_prior]`, gate +0.00409 P 0.999 | **ready, waiting for «да» (LB)** |
| Hedge of the second kurgan class, dual-class | tested | have/closed |
| Hillfort mask / neigh prior (EN +0.0003) | gormask −0.0056; neigh +0.0002 P 0.63 | **closed** |
| Li→Sp copy | −0.0037 | closed |
| Twins-only kurgans | kurgan_twins_only=true | have |
| EtGr layer isolation (EN +0.010) | ETGR_MAX_DEPTH_M=2.0, EtGr object_poly preserved | **have** |
| Pixel-GBM geophysics (EN +0.003…; the 0.348→0.412 jump on our side) | PIXGBM in eb268d6; clean-room E1G1 P1 PARITY PASS | have (clean-room in progress) |
| LGBM/GBM reranker over candidates | w5 V5, rerank_all | closed (kurgans) |
| Zoom-TTA satellite | EN −0.06 | closed |
| **Satellite: heterogeneous detector ensemble, mean-score at a low threshold (EN v12 +0.048)** | sp_focal + sp_focal2 + verify; sp_dfine off (unrepro), ORSP fusion <0.003 | **missing. The main lever, needs GPU** |
| **Satellite d4-TTA on spor_focal** (EN 0.326→0.339) | in the T7 queue, not measured | missing. Inference CPU is cheap, but OOF needs a predictor run → GPU/Kaggle |
| **LiDAR: height ×0.5–2.2 + zoom ×0.7–1.45 aug with relief recomputed per crop** | **not in our training code** (rg: none) | **missing. Needs GPU** |
| LiDAR: site-balanced sampling (EN +0.012–0.016) | not found in our training scripts | missing → retrain on GPU |
| LiDAR multi-seed blend (seed spread 0.545–0.569) / Family «2 of 4» consensus | single models + li_mosaic/li_dev | partial → GPU |
| Ae fortifications via FCOS (EN +0.003) | ae_fort group in the stack | partial; Δ in the noise |
| DINOv3 hillfort localiser (EN v23/v24, +0.03) | **DINOv3 banned**; a DINOv2-L alternative (T8) is not done | ban / GPU |
| Monte-Carlo model selection by stability (Family) | honest-10 + paired bootstrap P | have, as a method |
| Previous-stage weights (D-FINE/RT-DETR, EN legacy) | — | **cannot use** |

## 4. Adaptation plan, ranked by expected Δ

| # | Action (own implementation, idea only) | Expected LB Δ | Resources | Needs «да» |
|---|---|---|---|---|
| 1 | **LiDAR retrain: height ×0.5–2.2 + zoom ×0.7–1.45, relief channels recomputed after the transform, site-balanced sampler, 3 seeds → average heatmaps.** Gate honest-10 + new34 | +0.02…+0.05 (EN LiDAR R 0.676 vs ours 0.504) | GPU, ~1–2 days | **yes (GPU)** |
| 2 | **Satellite: a second, different detector** (not the R18-focal family: e.g. an anchor-free FCOS or CenterNet heatmap on our own torchvision/timm backbone, without YOLO/DINOv3) → mean-score fusion at radius 8 m, one threshold; OOF on the same folds | +0.02…+0.05 (sat R 0.255 → 0.4–0.6) | GPU, 1–2 days | **yes (GPU)** |
| 3 | Satellite d4-TTA spor_focal: OOF prediction with TTA → gate | +0.003…+0.01 | GPU/Kaggle for a few hours (inference) | **yes (GPU)** |
| 4 | FR1h on LB (already gated) | +0.004 | upload | **yes (LB)** |
| 5 | Hillfort localiser without DINOv3 (DINOv2-L/ConvNeXt, T8) | +0.005…+0.02 | GPU | yes |
| 6 | Ae fortifications (separate detector) | ≤+0.003 | GPU | yes, low priority |

What we start now without GO: nothing new from items 1–6. The CPU part is finished (sections 2.1 and 2.2), and the conclusion holds: **the CPU levers EN used are already either in eb268d6 or waiting for «да» (FR1h).** What's needed from Elena: GO for GPU on #1 and #2, «да» for the FR1h upload, and an answer on rule §0.

## 5. Do not repeat (closed at our end or at EN's)
WSR / ring / peak / dino-gor / sp_large / r50c5 / sat 8 m cluster / finds snap / gormask / neigh / Li→Sp copy / zoom-TTA / Ae kurgans / raw LAS returns / WiSE / LSKNet / DINOv3 (ban) / partial-label pseudo-labels / thresholds tuned on public / dome / slope_gate.
