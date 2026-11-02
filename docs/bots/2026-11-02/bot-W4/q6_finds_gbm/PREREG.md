# PREREG W4 Q6: candidate GBM for finds_points (clean room), gate over v3_on b447649

Committed 02.11 MSK **before any measurement**. Spec: orch/COMPETITOR_GAPS_2026-11-02.md §3 Q6 (bots/orch-protocol d2fe1de).
Clean room: no EN code, no expds/geo_pixgbm.py / geo_finds_gbm.py / models/pixgbm / bot3 branches (see orch/bot-W4/CLEANROOM_READLOG.md).
Code: own, on top of the allowed expds/geophys.py helpers (read_band, robust_z, highpass, accel.find_peaks, select_rasters,
survey_boundary/load_mask/sane_masks/apply_masks, circle_3857). Box CPU, nice 19, 4 threads.

## Candidates (fixed)
* Rasters: exactly the production classic selection `geophys.select_rasters(maggr files, "maggr", "gray")`; band mode, CRS, masks as classic
  (`[geophys.maggr.finds_points]` of configs/v3_on.toml). Decimation by block mean k = max(1, round(0.08 / res)) (200: k=2, 212: k=4, else 1).
* Detrend: hp = highpass(robust_z(a), res, 2.0 m) (= classic background_m 2.0).
* Candidates: local maxima of |hp| with min distance 0.5 m, |hp| ≥ 0.88 quantile of finite |hp| of that raster.
* Output: circle r 0.5 m at the candidate, class finds_points, markup_type maggr, threshold_group `maggr_fgbm`, confidence = OOF GBM probability;
  greedy NMS 1.0 m in confidence order; classic boundary/noise masks applied after (as classic).

## Descriptors (rotation-invariant, on hp)
centre hp, |hp|; ring means of hp and |hp| for r = 0.2, 0.4, …, 1.2 m (annulus ±0.1 m); disk r ≤ 1.2 m max/min/range/std;
8 sectors on annulus 0.2–0.8 m: opposite-sector products (min, mean of 4), max |s_k − s_{k+4}|, asymmetry |Σ s_k e^{iθ_k}| / Σ|s_k|;
raster quantile rank of |hp|; number of other candidates within 2 m; log res.

## Model
LightGBM binary, fixed params: num_leaves 31, learning_rate 0.05, 400 rounds, min_child_samples 50, feature_fraction 0.8,
bagging_fraction 0.8 / freq 1, lambda_l2 1.0, seed 0, 4 threads. Label: nearest GT finds point ≤ 1.0 m (UTM metres).
Folds = honest-10 folds of t_taus.json (site-grouped): predictions for a region come from a model trained on all finds-GT regions
NOT in that region's fold (fold 1 Гнёздово 208–212, fold 4 Семибратнее 201–207, fold 9 Судьбищи 200). 211 is not excluded.

## Variants (m = 2 primary)
* **REP** — replace: classic maggr finds dropped (their τ = 1.01 in every fold), GBM finds added.
* **UNI** — union: classic kept at base τ, GBM finds added as own group.

## Thresholds (Q0 lesson)
* **Primary = frozen**: every existing group keeps its base per-fold τ (t_taus.json taus_V3P). Only τ(maggr_fgbm, finds_points) is chosen
  nested: per held fold, argmax of the class-weighted score on the 9 training folds (gate curves, honest), grid
  0.05…0.95 step 0.05; tie → higher τ.
* Secondary (reported, not gated for the decision): full nested re-tune of all groups (D.honest), as in earlier gates.
* Sanity: UNI with τ = ∞ must reproduce gate base V3P 0.3337867 to 1e-6; production replay base must reproduce 0.345439.

## Gate (production replay, compute_metrics.py of the gate)
Replay = W1 satrec s2_replay protocol (archive b447649 code, v3_on.toml, watchdog.finish per fold) with the variant's taus/extras;
scoring = s3_score protocol. PASS iff all: Δall ≥ +0.003; paired bootstrap (site_family, 2000, seed 0) P(Δ>0) ≥ 0.95
(Bonferroni m = 2: 1 − 0.1/2); per-class floors kp/kt/gor −0.002, others −0.005; Δnew34 ≥ 0; Δnolidar ≥ 0.
The variant that passes with the larger Δall is the candidate; if none passes → FAIL. No LB submissions; flags default off;
production hook only if PASS (separate commit).
