# P-W4-compgate + v3_on+M_rp — result (30.10.2026 16:21 MSK), PREREG docs/PREREG_w4_compgate.md (c16be2f, before measurement)
R  rebuild A on box (fresh split+cache): 60/60 cluster blobs equal cached, honest 0.3146153227 → PASS.
Sanity: C0 Σ_mod counts == unweighted (0 mismatched cells, all variants); C1 |w≡1 − honest| ≤ 1.1e-16; M0 extras-without-disk == A and == V3P (0.0).
Weights (target/OOF GT share): PUB geo 1.19 li 1.49 sp_or 0.89 ae 0.13 or 1.03; CLO geo 0.48 li 0.90 sp_or 1.37 ae 1.01 or 11.9 (noisy).
| comparison | UNW Δ (P) | new34 | nolidar | PUB Δ (P) | CLO Δ (P) | comp-OK |
|---|---|---|---|---|---|---|
| V3P vs B0 | +0.019820 (0.976) | +0.008405 | +0.075076 | +0.014806 (0.942) | +0.019615 (0.976) | yes |
| MRP vs A | +0.004670 (0.9925) | +0.002431 | +0.008924 | +0.002353 (0.933) | +0.005075 (0.999) | yes |
| V3P+MRP vs V3P (gate) | +0.002758 (0.9985) | +0.003660 | +0.002369 | +0.001078 (0.998) | +0.002604 (0.998) | yes |
| V3P+MRP vs B0 | +0.022578 (0.9855) | +0.012064 | +0.077446 | +0.015884 (0.948) | +0.022219 (0.986) | yes |
V3P+MRP gate: Δ<0.003 → FAIL by prereg (P, floors, slices, composition all OK); per prereg: M_rp partly overlaps v3_on (+0.00467 on A → +0.00276 on top of v3_on).
MRP on A reproduces W3 +0.004670 exactly. Disclosures: TP attributed to predicted modality; group etgr_org (C19 extras) mapped to geo via the etgr prefix rule (not named in PREREG list).
