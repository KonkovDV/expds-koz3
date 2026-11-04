# PREREG WS-K1 (W3) — kurgan povr/tsel tie-break, written 30.10 01:30 MSK BEFORE any measurement
Data: ds/kcrops/kurgan_subclass_crops.npz (0.5 m, 64 px; lrm_s, lrm_l, slope; uint8), OOF folds from cpu/kurgan/work/kurgan_subclass_oof.csv (family folds).
Pre-check finding (not a gate number): split_arch kurgan conf ≥0.5 has 0 equal-conf povr/tsel twins out of 714 twin locations, so "ties" are near-ties, not exact ties.
Features t (crop is 32 m, so r10/r20 m → r3/r5 m disc vs 6–8 m ring): for lrm_s and lrm_l: peak−ring median, disc mean−ring median, prominence (peak−ring p10);
LRM blob circularity 4πA/P² (lrm_s > ring_med + 0.5·prom, component at the peak), fill fraction of the r5 disc, mean slope in disc.
Model: standardized LogisticRegression(C=1) trained on labeled rows (gt+cand, y=1 povr) of the other folds → OOF p_geo for every row. Diagnostic: AUC p_geo on cand.
Score (covered candidate rows only): z = clip(logit(p),−3,3)/3; povr copy conf+ε·z, tsel copy conf−ε·z, clipped to [0,1]. Nothing dropped or added beyond replacement.
Variants: KT1 p=p_geo, ε=1e-3 | KT2 p=σ(mean(logit p_geo, logit p_povr_ens)), ε=1e-3 | KT3 = KT2 with ε=1e-2 (sensitivity).
Seed average (41/42/2): seed kernels not finished → KS variants later under the same rule.
Gate: standard w3_gate (honest-10 Δ≥+0.003, P≥0.9, 34new and optical not below A=0.314615). Adopt only if pass; flag default off.

## Addendum KS (30.10 14:20 MSK, BEFORE measuring)
Seed average: mean of logit(p_povr) over all aligned kurgan OOFs available now: base 7867, out/w3-kurgan-cpu-seeds (s11/12), cpu/kurgan_s61,s63,s65,s67,s71,s73 (8 ensembles). Later seeds are not waited for.
Variants: KS1–KS5 = make_k.py rules on p_avg (prefix KS); KST = tie-break rule (z from p_avg only, ε=1e-3). Diagnostic: AUC p_avg on cand. Standard gate.
