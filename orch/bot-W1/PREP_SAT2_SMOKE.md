# SAT2 CPU SMOKE — **PASS** (bot-D/W1, 01.11 11:54 MSK; CPU only, no lease/GPU/LB; hold 0.41238)
- Venv: /workspace/koz3/venv_sat2 (torch 2.15.0+cpu, torchvision 0.29.2, timm 1.0.30, smp 0.6.0, rasterio 1.5.2). No ultralytics/YOLO/DINOv3.
- centerhead tests (wt/sp-large-f0): **50 passed, 1 skipped** (skip = no official compute_metrics.py).
- Smoke: orch/bot-W1/sat2_smoke/smoke_sat2.py → smoke_result.json. Backbone **tu-convnext_tiny + ImageNet** loaded (no R50 fallback), UnetCenterNet 32.0 M params.
- Data: 4 GRID windows 512 px overlap 0.25 (tile_origins) from fold0 tif 130_Курганы_1 (EPSG:32644, 0.85 m native), GT sp_or kurgans 20/15/15/14 per tile.
- 3 AdamW steps CPU: loss_center 5.217 → 4.863 → 4.526 (finite, falling); out hm (4,2,128,128), wh (4,1,..), off (4,2,..), mask (4,3,512,512).
- Decode → 3016 candidates → candidates_to_feats EPSG:3857 (markup sp_or) OK. ~85 s CPU.
- Caveats: smoke used native res (not resampled to 0.5 m); centerhead DEFAULTS markup_type "li", circle_r_m 0.5, cls order tselye=0/povr=1 → set for sp_det2; seg 'mask' head unused (kurgans only).
- Not done (needs code, CPU OK): sat grid tile builder/Dataset for centerhead (expds.tiles grid cache → hm targets), pinned split, N-group sp_score_fusion.
- Needs lease (A): fold0 training kernel …/koz3-sat2-f0 + DS …/koz3-sat2-f0-data upload (~1 GB grid f0 or raw f0 2.1 GB).
