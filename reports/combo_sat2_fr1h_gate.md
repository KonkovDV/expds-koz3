# Combo gate: sat2 + FR1h (honest-10 / gate_v6c)

A=0.314615  B′=0.315800  boot=4000

| variant | score | ΔA | ΔB′ | P(gt0) | p5 | damaged | pass_full |
|---------|-------|-----|------|--------|-----|---------|-----------|
| sat2 REPLACE scale 1.0 | 0.326450 | +0.011835 | +0.010650 | 1.0 | +0.00477 | 0.03429 | yes |
| sat2+FR1h | 0.325626 | +0.011011 | +0.009826 | 1.0 | +0.00427 | 0.03178 | yes |

**Chosen: sat2_only** (sat2+FR1h < sat2 alone). FR1h dropped from pack.

## Submission
- path: `/workspace/koz3/out/combo/submission_sat2_on_eb268d6.zip`
- sha256: `d85e9b2f3c63090583f11a2ba03fa2a1427f47b6a1838a3a8cd959ad5fe5b73b` (~1069 MiB)
- base: FR1h-on-eb268d6 tree stripped of `[post.frame_prior]` + archive thr restored
- + `expds/sp_det2.py`, patched `pipeline.py` hooks (FR1h-base, no lidar_cnx import), `centerhead/`, 5× weights, `[sp_det2] enabled=true`
- default.toml in repo keeps `[sp_det2]` commented OFF; profile `sat2` enables via apply_release_profile
- Kaggle check: `dmitrijkv/koz3-w1-sat2-check` (dataset `dmitrijkv/koz3-sat2-chk-20261101`)
- Drive: folder `17N5Cu1bZJ38iYu3W_nCN3zOk6R3l5riX` parts `.00`–`.11` + SHA256SUMS + README

## Smoke (Kaggle CPU, pantum/n7q check_submission)
- kernel: dmitrijkv/koz3-w1-sat2-check v2 — **PASS** FAIL0 WARN1, wall 81s
- main.py 22.3s, peak RSS 1875 MiB; phases include **sp_det2: сырых 5, 8 с**; written 227
- sha_ok=True
