# KURGAN-DUAL PREREG — W3, 02.11 MSK. Committed before any measurement.
Source: orch/COMPETITORS_2026-11-02.md §3 #2 (bots/orch-protocol b2abce3); task from bot 1, approved by Elena 02.11. Clean-room: written from this spec only, no EN code.
## Context (prod)
Base B = V3P (v3_on b447649 in stack form, as W4 compgate.py: stack without etgr and without sp rows in D-FINE regions, plus C19 etgr and D-FINE sp extras), honest-10 family folds, d3_gate TableEngine/honest unchanged. Expected B = 0.333787.
Fact (checked before PREREG, label-free): every lidar kurgan peak in the stack (groups li_dev, li_mosaic, li) exists as two rows with identical geometry, one for kurgany_povrezhdennye and one for kurgany_tselye, each with its own conf. In B each copy passes only its own class×group threshold.
## Single variant DUAL (flag kurgan_dual, default false)
For each li pair, the second class = the copy with the lower conf (ties: tselye is second).
In outer fold k, B thresholds τ[g][c] come from the standard train sweep (D.honest).
A second-class copy c that B does not emit (conf_c < τ[g][c]) is additionally emitted if:
- conf_c ≥ τ_dual, and
- its partner (the first class) is emitted in B (conf_other ≥ τ[g][other]).
τ_dual is one value per outer fold, shared by both classes and all li groups. The grid is {off, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20}.
τ_dual is chosen on train regions of fold k only. Criterion: score_counts of the two kurgan classes with τ[k] applied. Ties go to the larger τ (off counts as largest).
Hit flags: the standard curve flags (matching over all candidates, as in every gate).
Other modalities (sp/ae twins already in the pipeline) are unchanged.
## Gate DUAL vs B (honest-10, paired family bootstrap, 2000)
Bonferroni: 1 variant, so P ≥ 0.9.
Accept if all hold:
- Δ_all ≥ +0.003;
- Δ new34 ≥ 0;
- ΔF1 kurgany_povrezhdennye ≥ 0;
- ΔF1 fortifikatsii ≥ 0;
- every class meets the W4 floors (≥ −0.005; tight classes −0.002);
- nolidar Δ ≥ 0.
Report Δ, P, the per-group deltas, and τ_dual per fold.
