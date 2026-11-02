# BEAT_EN_WORLD_PLAN_20261101 — EN (ideas only) + world SOTA → BEAT_EN_PLUS plan

A-executor, 01.11 14:27 MSK (UTC+3). Elena t245u + t246u. **Ideas only.** Sources were EN docs, ledger and commit messages, already read in
`plans/EN_COMMIT_GAP_VS_US_20261101.md` and `plans/EN_PLUS_WORLD_SOTA_20261101.md`. No EN .py or weights were opened or copied this pass.
Rules: hold LB 0.41238 (bot never submits); no main; no default.toml change; closed levers stay closed
(WSR, ring, peak, dino-gor, sp_large, r50c5, sat 8 m cluster, sat fusion arms).

## 0. Where we are vs EN (numbers)
| | LB | Notes |
|---|---|---|
| EN v22 | **0.50491** | +0.088 over us |
| us (eb268d6) | **0.41238** | geo .769/.656 ≈ EN; lidar P/R .516/.504; sat .456/.255 |
| EN's two model jumps | v11 +0.023 (LiDAR augx blend), v12 +0.048 (heterogeneous sat fusion + framing) | everything after v14 is ≤ +0.01 each, mostly geo/Ae, which we have |
| Gap model | LiDAR→EN ≈ **+0.053**, sat→EN ≈ **+0.051** | the whole gap is kurgan recall in two modalities |

## 1. What EN did that matters (one line each) → our status today
| EN lever | Our status 01.11 14:30 |
|---|---|
| LiDAR relief recomputed after DTM height/zoom aug + site balance + openness±/SVF + recipe blend | **R-augx v2 RUNNING** (elena gpu-1, e514195: DTM banks + α0.5 + c7 opn_r/msrm), fold0 result pending |
| Heterogeneous satellite detectors (DETR family + heatmap), mean fusion, absent = 0 | det2 CenterNet + ConvNeXt-T fold0: standalone damaged F1 **0.4175 vs 0.3677**. All 4 fuse arms FAIL. Replace honest-10 FAIL as a gate (1 family), but fold0-local +0.067 (`out/sat2-f0/REPLACE_HONEST.md`) |
| Framing prior (d_rel) | FR1h pack ready, OOF +0.00409 P 0.999. **Waits «да»** |
| LiDAR neighbourhood prior in decoder | code exists, off; only on the new heatmap |
| PIXGBM geo | ours is EN-line in 0.41238 (DOC-1 risk). **Clean-room E1_G1 zip built today** (FR1h + clean pix, lleaves) |
| DINOv3 hillfort localiser | banned for us |

## 2. World SOTA / real cases (2019–2026). New this pass: rows W18–W22; W1–W17 are in EN_PLUS_WORLD_SOTA §2
| # | Source | Finding | Transfer |
|---|---|---|---|
| W1 | Berganzo-Besga 2021 *Remote Sens.* 13:4181 (MSRM + YOLOv3, 10.5k tumuli, NW Iberia) | MSRM beat slope/SLRM (AP 63 %), DTM raw AP 21.8 %; P .97 R .64 after an S2 soil mask | MSRM channel = in R-augx (c7) ✔ |
| W6 | Verschoof-van der Vaart, WODAN2.0, *ISPRS IJGI* 2020 | curated test ~70 % → random test ~50 % for barrows; bagging + location-based ranking +17–35 % | our honest-10 / new34 = the "random" view; LBR ≈ framing prior (FR1h) |
| W8 | Chen F. 2021 ISPRS J; 2026 *Remote Sens.* 18:185 (Altai kurgans, GF-2 0.8 m) | focal-type imbalance losses + larger backbones raise every detector | det2 (ConvNeXt-T + focal) is this recipe; next: ConvNeXt-S, explicit background windows |
| W18 | **Shang et al., L-CenterNet, IGARSS 2025** (doi 10.1109/IGARSS55030.2025.11243009) | CenterNet for tiny aerial objects: point-supervised inverse-square distance maps (ISDT) instead of Gaussian; NWD loss instead of L1 box regression | det2 v2 head: ISDT target + NWD size loss for small mounds. Cheap, same trainer. Candidate for folds 1–4 *only if* fold0 re-check |
| W19 | **Europe-wide hillfort CNN, JCAA 2025** (10.5334/jcaa.178) | England-trained CNN fails abroad ("model drift"); fine-tuning on a small local sample + pseudo-labels → F1 0.34–0.38 | confirms why per-family honest tau matters. **Test-time self-training is NOT adopted**: EN measured it as negative (0.564→0.498) |
| W20 | Verschoof-van der Vaart, WODAN in practice, S. Netherlands (2025, Augsburg OPUS 124213) | transfer to new geology still finds barrows, but needs human-in-loop | none for scoring; supports conservative tau |
| W21 | Sobotkova et al. 2024, *J. Documentation* 80(5) (Bulgaria field validation) | tile CNN FN 95 %; training only on "highly visible" mounds was worse | keep faint/damaged in det2 train (we do) |
| W22 | CORONA / Mesopotamia, *PLOS ONE* 2025 (10.1371/journal.pone.0330419) | retraining an existing site model on a new image domain → IoU > 85 % | domain-specific retrain > generic. Our sat tiles mix 0.1–3 m GSD → GSD-bucketed aug (S+2) |

Net: every real case says (1) relief representation (MSRM / multi-scale) for LiDAR mounds, (2) bigger backbone + imbalance loss for satellite mounds,
(3) heterogeneity / bagging over seeds, and (4) random-geography evaluation is 20 pt harsher than curated. We have honest-10 and must keep it.

## 3. BEAT_EN_PLUS — plan (ordered; each step = gate; Elena «да» where marked)
| # | Step | Compute | Exp. Δ (honest → LB) | Gate / kill | Needs |
|---|---|---|---|---|---|
| 0 | **Insurance**: non-final LB probe of clean zip `out/geo-clean-pack/submission_pixgbm_clean_e1g1_p4_lleaves.zip` (FR1h + clean PIXGBM) | 0 | DOC-1 floor 0.378 → ≈0.41–0.42; FR1h +0.004 | LB ≥ 0.41238 − 0.003, else keep archive | **Elena uploads** |
| 1 | LiDAR R-augx v2 fold0 (running) → if fold0 li-only ≥ c3-sc baseline +0.01: folds 1–4 (≈4×2 h, 1 GPU) → **replace li-kurgan** honest-10 | GPU | +0.01…+0.025 | prereg P-W2-LA; replace gate Δ≥+0.003 P≥0.9; k*≥5; thr-dispersion ≤×2 | lease (have), folds «да» |
| 2 | **SAT det2 folds 1–4** (same recipe, dmitrijkv/gpu-1, ~4×45 min) → det2 replaces sp_focal everywhere with its own group tau → honest-10 | GPU 3 h | +0.005…+0.015 | standalone F1 > spor_focal on ≥3/4 folds; replace gate as above | **«да»** (GPU held today) |
| 3 | LiDAR recipe blend (R-aug, R-augx, R-ms × 1 seed) + neighbour prior on new heatmap | GPU 2 slots | +0.005…+0.015 | blend > best single on li-nested | after #1 |
| 4 | SAT heterogeneity: 2nd *different* family next to det2 (ConvNeXt-S CenterNet with ISDT/NWD (W18), or D-FINE no-DN own-trained if licence OK), mean fusion of **det2 + new** (not sp_focal) | GPU | +0.005…+0.02 | fusion beats det2 alone on fold0, then honest | after #2 |
| 5 | FR1h+M_rp pack LB | 0 | +0.004…+0.009 (OOF) | LB | **Elena** |
| 6 | Runtime rehearsal of the full new stack ≤ 1500 s budget (pix 900 + lidar blend + det2) | CPU/Docker | 0 (enabler) | docker smoke, 0 classic | before any pack |

Ceiling if 1+2+3+4 transfer at the midpoint: ≈ 0.41 + 0.018 + 0.010 + 0.010 + 0.010 ≈ **0.46**. Beating EN 0.505 needs the upper ends of #1 and #4 together. **Honest P(>EN) is low (<15 %).**

## 4. Beyond EN (things EN did not do)
- LiDAR: discrete DTM banks (parity-exact with inference); MSRM channel; α=0.5 site temperature; recipe>seed; thr-dispersion kill. These are already in R-augx.
- SAT: **replacement**, not mean-fusion, when one member dominates. Today's 4-arm FAIL + fold0 replace +0.067 show this. EN averaged correlated members.
- SAT: ISDT/NWD CenterNet target for tiny mounds (W18). EN never used CenterNet-style point supervision.
- Eval: replace-gates must be **all-fold OOF**. One-family replacement contaminates the shared tau (optical −0.0048 today). This is a protocol rule now.
