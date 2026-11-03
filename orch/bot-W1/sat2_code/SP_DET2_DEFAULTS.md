# sp_det2 — centerhead DEFAULTS overrides (bot-D/W1, 01.11 MSK). Side only; default.toml untouched. Hold 0.41238.
centerhead/config.py DEFAULTS are LiDAR-oriented (bot-9). For sat sp_det2 use `cfg_with(...)`:

| key | DEFAULTS | sp_det2 | why |
|---|---|---|---|
| markup_type | "li" | **"sp_or"** | export feats as sat markup; eval/metric filter by markup sp_or |
| circle_r_m | 0.5 | **4.0** | match spor_focal `--center-radius 4` (sp_or circles); sweep 2/4/6 on OOF later (official matching = centre-in-polygon/IoU) |
| min_score | 0.05 | **0.03** | raw candidates for fusion/nested thr (gen_floor), filtered later |
| res | 0.5 | 0.5 (grid cache res) | spor_focal is 1 m/t128 → heterogeneity (PROMPT EN #22) |
| crop | 512 | 512 | grid tile 512 px = 256 m |
| stride | 4 | 4 | 2 m output cell; kurgans 5–40 px at 0.5 m |
| sigma_mode/min_overlap | centernet/0.7 | same | |
| default_diam_m | 8.0 | same | sp_or kurgans are polygons → D = 2√(S/π) |
| nms_min_m / nms_frac / nms_cross_class | 1.0/0.8/False | same | cross-tile dups handled by own-cell rule (nearest window centre, Chebyshev) |
| focal α/β, w_hm/w_wh/w_off | 2/4, 1/0.1/1 | same | |

**Class order:** network heatmap channels follow `HM_CLASSES` = (`kurgany_tselye`=0, `kurgany_povrezhdennye`=1).
expds.tiles cache mask bits follow `RS_CLASSES` filter = (povr=bit0, tselye=bit1) — opposite! `grid_labels.py` builds objs from GT
with HM_CLASSES (verified: 1787/1799 train object centres fall on the matching mask bit within ±2 px; 12 misses = tiny/edge polygons).
**Seg 'mask' head** (gor/fort/arch) is unused for kurgans-only; its output is ignored (no loss).
**Input norm:** ImageNet mean/std on uint8 (cache band_stats stretch = same as inference `to_uint8`).
**Folds:** pinned spor_focal family folds from `sat2_dataset_manifest.json` (fold0 = family «Курганы», 34 regions) — NOT recomputed from grid positives (15 families would move).
**threshold_group:** "sp_det2"; thresholds placeholder 1.000001 (emit nothing) until nested k10.
