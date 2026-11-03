# ENS6 full-train E1_G1 (NOT L) — bot D / W3 — 31.10 01:30 MSK
- Kaggle: NO ens6 kernel pushed. Each full-train fit took ~3 min on the box (nice10, 3 thr); a Kaggle push needs a 0.6–2.2 GB upload per account. aaaa0000004 left free for B GPU 03:05.
- Full-train ALL_s0 DONE 01:04 (w4geo/work/): models_{n40r8,n40r6,n40r10,aug,gw1d,gw3d}full/ALL_s0. These are in-sample on train; honest numbers = OOF gate.
- Frozen E1_G1 (sha c92b25e0, untouched), OOF gate vs A18:
  - raw: geo P 0.3564 · R 0.7976 · Δ +0.0407 (finds +0.179, poly +0.228) · P(better) 1.0
  - _ht: geo P 0.6925 · R 0.6738 · Δ +0.0425 (finds +0.181, poly +0.245) · P(better) 1.0 · PASS_vs_ref False (MagGr R −0.162)
- Seed-ens through G1 recipe (cstar/C_CetO3_Sr8G_noTerFl_L_E1_G1R_*):
  - bens5 _ht: P 0.5541 · R 0.6810 · Δ +0.0340 · P(better vs A) 0.97 · vs G1_ht −0.0086, P 0.0 → STOP
  - e6b5 _ht: P 0.5863 · R 0.7007 · Δ +0.0375 · P(better vs A) 0.97 · vs G1_ht −0.0050, P 0.0–0.006 → STOP
  - (raw P 0.249 / 0.260)
- Verdict: keep frozen E1_G1. Seed-ens does not beat it. Files: geo5_seedens/R_G1R.json, R_G1R_vsG1ht.json.

---
## FINAL night geo step (A→D 31.10 ~01:31 MSK)
- STOP rule: candidacy uses **_ht** (Elena P≈0.7 band). Raw P~0.36 = STOP for LB-probe shape only.
- E1_G1_ht stays frozen; seed-ens no beat → **no C re-gate**.
- Do **not** push more ens6 Kaggle unless MagGr-13 full weights packaging later — local ALL_s0×6 enough for night measurement.
- aaaa GPU kept free for 03:05 r50c5. No LB.
- Status: **FINAL** for D night geo.
