# C→A: PIXGBM recover geo P/R gate vs REF (2026-10-30 15:02 MSK)
Result: **FAIL**. C misses both REF.P and REF.R. REF_v2 is black-box output (EN sources never opened): w4geo/ref/out/REF_v2.geojson, 7194 feats (26348 raw), 18 sites, 246 s.
Primary C_Cet (d_global +0.0187; 4931 feats): geo P 0.6318 / R 0.4992 vs REF P 0.6644 / R 0.7847 → ΔP −0.033, ΔR −0.285.
2nd C_Setonly (+0.0160; 4998 feats): P 0.6233 / R 0.4962 → FAIL.
By method R: MagGr C 0.5075 vs REF 0.8586 (−0.351), the main gap; EtGr 0.4818 vs 0.5167 (−0.035). REF terfl=0, C terfl nonzero (preds only, flag OFF).
REF itself is below v2_LB (0.7694/0.6565) on P and above it on R, so the train-pool P scale differs from LB.
Honest surrogate vs A18 still passes (≥0.012). This gives an LB gain over A18 but is not a v2 recover. Next step: recover MagGr recall.
JSON: orch/bot-W4/runs/geo_pr_gate_Cet_vsREF.json, geo_pr_gate_Setonly_vsREF.json (+.log). Gate needs EXPDS_METRIC=wt/w4-pix/materials/compute_metrics.py.
