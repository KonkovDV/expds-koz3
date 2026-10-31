# F:\expds3 inventory after 10:40 MSK move (bot 1)
Top level: work 68.2 GB (was 114.2), TRASH 46.0 GB, mock_val 11.28 GB, organizers 0.07 GB, _s3_listing.csv, empty eval_geo/eval_gnezdovo/mock_geo.
Full CSVs on PC: C:\Sloi\pc_jobs\expds3_trash_out\{inventory.csv, inventory_top.csv, refcheck.csv}.

## KEEP (base A / v2 / active)
cache 11.27, li_fort_mask 8.43, spor_gor_wide 4.26, surf_arch 3.84, fusion 3.49, li_plus_oof 2.49, li_hard2_oof 2.29, ae_fort_mask 1.71, li_gor_mask 1.60,
spor_focal_oof 1.28 (= sp_focal2 of base A), sp_arch_oof 1.22, stack_verify_sp / stack_base / stack_verify_sp_arch / stack_arch geojsons (~4.4), mosaic_li 0.35,
fold0..fold4 (1.7), models 0.11, gt_train*.geojson, surf_arch_all.pt, sp_crop.pt, all *_oof.geojson layers, thresholds/score json, small scripts/logs.

## UNCERTAIN (live refs found in scripts, not moved) ~9.7 GB
li_hard3_oof 2.12, li_hard3_nocp_oof 2.12 (run_hard3*.py in v0.2.0 / wt_bot7 / wt_sim / wt_b10 analysis);
spor_r1_oof 1.40 (97 refs: rescore_rank, diag_r1_geom...); spor_all_oof 1.31 (run_spor_after, score_joint_spor); spor_center_oof 1.27 (audit_optical_gt);
wheels 0.16 (scripts/build_rust.py - needed); smoke_in/smoke_coarse/smoke_dock/smoke_win/bench_in/bench_mix/mock_out/__pycache__ (~0.05, referenced by analysis/test scripts).
All refs are in analysis/* copies of old experiments, none in expds/ pipeline; can be moved later if owner OKs dropping those analysis repros.

## Loose files (~6 GB, not moved)
Old-experiment geojsons (li_*_oof.geojson 0.45 each, alldata_dev* 1.4, tin_*.npy 0.8, ch_*.npz) - layer outputs, small, kept as history.
