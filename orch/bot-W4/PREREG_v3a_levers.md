# PREREG v3a CPU levers on OOF A (bot C, 30.10.2026 13:14 MSK — written BEFORE any measurement)

Data: w5data/stack_arch.geojson -> work/table.pkl (OOF A), gt_train_v2. Base A = honest-10 0.3146153226828423 (103 families).
Protocol = gate_v6c / w5_policies: per-(class, threshold_group) thresholds swept on train folds (honest_counts, unseen=drop),
paired family bootstrap 2000 vs base_counts. PASS: d>=+0.003 AND P(d>0)>=0.9 AND d_new34>=-0.001 AND d_optical>=-0.001,
per-class dF1 floors: -0.002 kurgany_povr/tselye/gorodishcha, -0.005 others. Param grids fixed below; reported number =
`*_nested` (param picked per outer fold by inner 9-fold honest score). Fixed-param rows are reference only.
Bonferroni note: 5 levers -> P>=0.98 noted (not used for PASS, per coord rule P>=0.9).
All levers are candidate/score/group transforms usable at test time (site layers = inputs present; geometry = own preds).
Code: orch/bot-W4/v3a_levers.py (flag-free measure-only transform on OOF; archive code untouched, default off).

## Already measured & rejected (not re-run, no new data) — W5 real_policies_V0-V4.json (bots/w5-rerank 502bb3b)
- #4 cross-layer copy Li->other layers (V3): nested d=-0.0098 P0.00. Structural reason (diagnostic 13:20): official
  matching is region-scoped, NOT layer-scoped -> a copy can never add a hit an original would not get; only adds FP/thr noise.
  Oracle: of 1162 ae kurgany_povr GT, any candidate hits 523 (152 by ae preds, rest by li/sp_or) -> 639 ae povr GT have NO
  candidate at all => aerial recall is detector-bound (GPU), not post-proc-bound.
- #6 multihyp kurgan hedge (V2): nested -0.0132. #10 micro-circles (V4): nested -0.0042. min-k (V1) -0.0019.

## New levers (this PREREG)
| id | lever | transform | grid |
|---|---|---|---|
| L3m | #3 break-even τ, modality-gated: thresholds per class x group x site-modality | group += '|nl' if site has no li layer in the stack, ('|ae1' if ae is the only optical layer) | variants: {nl}, {nl, aeonly} |
| L3a | #3 large-object split (priority b): thresholds per class x group x area bin for gor/fort/object_poly/arch | group += area bin of pred polygon (<1k, 1k-5k, 5k-10k, >=10k m2) | bins sets: {5k}, {1k,5k,10k} |
| L10o | object_poly size filter (#10 part) | drop object_poly preds with max(bbox w,h) > X m | X in {30} primary; refs 20, 50 |
| L14 | same-class dedup across groups | within (site, class family) greedy by conf: drop pred whose bbox-IoU (centroid-centred bbox proxy) with a kept higher-conf pred of a DIFFERENT group >= t | t in {0.3, 0.5, 0.7} |
| L14s | same-group dedup | as L14 but SAME group | t in {0.5, 0.7} |
Not measurable on box (blocker): L8 EtGr depth >2 m (no depth attr in stack/table), L13 survey-contour clip (no contours on box).

Stacking: only PASS levers, composed in table order L10o -> L14 -> L3a/L3m groups; stack gated once (v3a_stack_gate.json).
Composition gate (#12, tool): every row also reports score re-weighted to public-LB modality mix (geophysics 0.50, lidar 0.33,
sp_or 0.127 (remainder, assumption), aerial 0.026, ortho 0.017; site modality = primary layer of its GT/preds); informational, not a PASS criterion.

## Addendum 30.10 13:20 MSK (BEFORE measuring these; L10o/L3m/L3a already measured 13:15–13:18 -> all FAIL, not stacked)
Composition gate #12 (tool, orch/bot-W4/scripts/compgate.py; also inline in v3a_levers.py):
- site modality = priority geo(maggr/etgr/terfl) > li > sp_or > ae > or over GT ∪ pred layers of the site.
- PUBLIC mix (PATH_TO_061 §1): geo 0.50, li 0.33, ae 0.026, or 0.017, sp_or 0.127 (remainder, assumption).
- CLOSED mix (assumption, stress for "ae/or, no lidar"): geo 0.25, li 0.10, sp_or 0.25, ae 0.25, or 0.15.
- Weighted score = score_counts(weights = mix[m]/train_share[m] per site); weighted paired family bootstrap 2000 -> P_pub, P_closed.
- Informational: gate PASS still = honest-10 rule; #12 reported as d_pub/P_pub, d_closed/P_closed. Required ≥0 for LB proposals (BREAKTHROUGH).

X1 xlayer (BREAKTHROUGH rank 1, REVIEW_SOTA #4/L1), directional, differs from rejected V3 (all->all, all candidates):
- copy a pred to each OTHER layer present at its site (pred layers), directions only Li->Ae, Li->Sp_or, Sp_or->Or, Or->Sp_or (no Ae->Li).
- only "confident" sources: conf >= q-quantile of its (class, group) on the whole table (test-time knowable rank), q in {0.5, 0.9};
  copy score = conf (β=1), own threshold_group `<grp>~x><layer>` swept on train folds (= break-even via F1-optimal τ).
- Grid q in {0.5, 0.9}; nested pick. Extra stop: FP on ae/or copies > +30 % without TP -> FAIL (reported as copy tp/fp).
- Prior expectation (written now): ≤0 — the official metric matches globally per class (compute_metrics: markup_type only for
  reference stats), so a same-geometry copy can only hit a GT its source could hit; gain only via separate τ of the copy group.

L3be break-even τ (REVIEW_SOTA #3): per outer fold, on train sites: isotonic p(hit|conf) per (class, group) (groups with <5 train hits
keep sweep τ), F_c = train sweep F1; keep iff p > F_c/(4−F_c). One variant, no grid.

L14 / L14s: as above; IoU on true UTM polygons if work/geom_wkb.pkl rebuilt in time, else bbox proxy (stated in JSON).
M_rp (v3a-minimum component): reproduce W0-seams NK_rp (5171c35 m1_anchor rp pairs for gor/arch/obj/fort) after geom rebuild;
v3a-min = A + M_rp (ORG3 lives in a separate 18-site geophysics OOF not on this box -> can only be reported by arithmetic, labelled).

## Addendum 13:27 MSK (before measuring L14g). Results so far: X1 FAIL −0.011; L3be +0.0015 P0.87 FAIL; L14 FAIL −0.019 (bbox-IoU proxy;
geom rebuild finished after launch) but site-modality split showed geo sites +0.0095 while li −0.029 -> one follow-up, pre-registered:
L14g = L14 (cross-group dedup, bbox-IoU proxy) restricted to class object_poly (geophysics layers only). t ∈ {0.3, 0.5, 0.7}, nested.
Same gate. Risk of selection on the split noted (1 extra test; Bonferroni ×2 -> P≥0.95 noted).
