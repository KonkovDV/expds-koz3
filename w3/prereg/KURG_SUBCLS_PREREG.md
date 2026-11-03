# KURG-SUBCLS PREREG — W3, 02.11 MSK. Committed before any measurement.
Task from bot 1, 09:31. One variant: SUBCLS (flag kurg_subcls, default off). Clean-room, CPU only, no EN code, no CNN.
## Data
Relief crops come from ds/kcrops/kurgan_subclass_crops.npz:
- 64×64 px at 0.5 m; channels lrm_s (±0.75 m), lrm_l (±2 m), slope; u8 with 128 = 0 m.
- Candidate crops have stack_idx = row of stack_arch, already checked: 6109/6109 match region, class and li.
These crops are a sample of li kurgan candidates. Li pairs without a crop stay unchanged (as in V3P). Their coverage of the li rows that V3P emits is logged as a diagnostic.
Labels:
- the npz y: the subclass of the nearest kurgan GT within 2 m;
- only labelled candidate crops are used, and only from train regions of the current fold.
## Features
Computed per crop in metres; the centre is the peak.
- Height: h = max lrm_l at r ≤ 4 m; h_s = max lrm_s at r ≤ 3 m.
- Volume: sum of positive lrm_l at r ≤ 8 m × 0.25 m².
- Symmetry: radial profile in 8 azimuth sectors (mean lrm_l, r = 1–8 m). Features: variance across sectors, max − min, coefficient of variation.
- Flatness: std of lrm_l at r ≤ 2 m divided by h; mean slope at r ≤ 2 m.
- Pit: max(lrm_s at r ≤ 4 m) − lrm_s at the centre (r ≤ 1 m); minimum lrm_s at r ≤ 2 m.
- Edge sharpness: max of the radial derivative of the mean lrm_l profile; mean and p90 slope in the ring 3–8 m.
- Ploughing anisotropy: structure-tensor coherence of the lrm_s gradient over the whole crop and over the ring 8–16 m.
- Confidences: conf_PO and conf_TS of the li twin pair (key reg, cx, cy, area, nv, grp from table.pkl), their difference, and the li group code.
## Model
LightGBM binary (y = 1 damaged):
- 200 trees, lr 0.05, num_leaves 15, min_child_samples 20, subsample 0.8 (freq 1), colsample 0.8, seed 0, n_jobs 4.
- No weights.
Nested:
- For outer honest-10 fold k, the model is trained on train-region labelled crops and predicts the held regions.
- For choosing m, train-region crops get inner-OOF p from a 9-fold split over the other outer folds.
Diagnostic: outer OOF AUC on labelled candidate crops (bot-9's earlier reference: CNN 0.707, relief 0.526).
## Rule for a li pair with a crop
- Pair score u = max(conf_PO, conf_TS).
- If |p − 0.5| < m, emit both copies with score u.
- Otherwise emit only the class with the higher p (p ≥ 0.5 means damaged), with score u; the other copy is removed.
- The rows go to group g+"|s", so the nested sweep picks their thresholds separately.
- m grid: {off (= V3P), 0, 0.05, 0.10, 0.20, 0.30, 0.50}.
- m is chosen per outer fold by the kurgan-class score_counts on train regions, with thresholds from the train sweep and train p inner-OOF. Ties prefer off, then the larger m.
- Curves are rebuilt with standard greedy matching in the new order. Held counts use the chosen m with the fold's train thresholds.
## Gate vs V3P 0.333787 (honest-10, paired family bootstrap 2000)
1 variant, so P ≥ 0.9.
Accept if all hold:
- Δ ≥ +0.003;
- new34 ≥ 0, nolidar ≥ 0;
- ΔF1 intact ≥ −0.002, ΔF1 damaged ≥ 0, ΔF1 fort ≥ 0;
- W4 class floors hold.
Report AUC, Δ, P, floors, and m per fold.
