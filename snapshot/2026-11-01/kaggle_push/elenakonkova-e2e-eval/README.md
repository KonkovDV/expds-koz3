# koz3-e2e-eval (elenakonkova, GPU T4, private)

Reusable mock-gate e2e for KOZ3 candidates. Screening on Kaggle; final confirmation in PC docker only for the release candidate.

## Attach
- `elenakonkova/koz3-mock-val-gate` (regions + gt.geojson)
- `elenakonkova/koz3-e2e-tools` (score + geo_pixgbm inject)
- `elenakonkova/koz3-e2e-baseline-v2` (default calib zip) OR a candidate dataset with `SPEC.json` + zip

## Request a run
1. Create private dataset `elenakonkova/koz3-e2e-cand-<id>` with `SPEC.json` and `candidate.zip` (or set zip name in SPEC).
2. Edit `kernel-metadata.json` dataset_sources to include that dataset (keep mock/tools).
3. Push: `KAGGLE_API_TOKEN=$(cat ~/.secrets/kaggle2_token) ~/.venvs/kaggle/bin/kaggle kernels push -p /workspace/kaggle/push/elenakonkova-e2e-eval`
4. Pull outputs: `e2e_eval/scores.json`, `summary.json`, `calibration.json`, `warn_scan.txt`, `STATUS.txt`

## Validity
exit==0, geojson present, wall_s >= 30, weights non-empty, pixgbm sha = 35e4331b…
