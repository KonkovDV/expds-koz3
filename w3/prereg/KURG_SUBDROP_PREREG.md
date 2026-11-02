# KURG-SUBDROP PREREG — W3, 02.11 MSK. Committed before measurement. Last attempt on the KURG-SUBCLS line (approved by bot 1, 09:43).
Same data, features, LightGBM, and nested outer/inner p as prereg/KURG_SUBCLS_PREREG.md (code/subcls.py). One variant; flag kurg_subdrop, default off.
Rule: for a li pair with a crop, if |p − 0.5| ≥ margin, remove the twin copy of the class the classifier disagrees with (p ≥ 0.5 → remove tselye, else remove povrezhdennye).
Each copy keeps its original conf and threshold group. Pairs without a crop and all other rows are unchanged.
Margin grid: {off, 0, 0.05, 0.10, 0.20, 0.30}. The margin is chosen per outer fold on train regions (kurgan score_counts, thresholds from the train sweep, inner-OOF p). Ties prefer off, then the larger margin.
Gate vs V3P 0.333787 (honest-10, 2000 bootstrap). Accept if all hold:
- Δ ≥ 0.003, P ≥ 0.9;
- new34 ≥ 0, nolidar ≥ 0;
- intact ≥ −0.002, damaged ≥ 0, fort ≥ 0;
- W4 class floors hold.
If FAIL, KURG-SUBCLS is closed.
