# Composition gate #12 (bot C, 30.10 13:29 MSK)
Implemented inline in orch/bot-W4/scripts/v3a_levers.py (extends the gate_v6c honest-10 protocol, no new framework):
- site modality: priority geo(maggr/etgr/terfl) > li > sp_or > ae > or over GT ∪ pred layers. OOF A sites: geo 18, li 58, sp_or 121, ae 9, or 4.
- PUBLIC mix (PATH_TO_061 §1): geo .50, li .33, sp_or .127 (remainder, assumption), ae .026, or .017.
  CLOSED mix (assumption, stress for "ae/or, no lidar"): geo .25, li .10, sp_or .25, ae .25, or .15.
- weighted score = score_counts(counts, weights = mix[m]/train_share[m] per site); P = weighted paired FAMILY bootstrap (1000).
- base A: honest 0.314615 -> public-weighted 0.359977. Every v3a_*.json row has d_comp/boot_comp (public), d_closed/boot_closed,
  d_by_site_modality (per-modality score deltas).
Smoke/real: M_rp: d +0.00437 P0.989 | public +0.0021 P0.92 | closed +0.0038 P0.999 (runs/v3a_ID_Mrp.json).
Caveats: ae (9 sites) and or (4 sites) carry huge weights in CLOSED -> high variance; mixes are assumptions until organisers' composition known.
Usage: cd wt/w5/analysis/2026-10-30/w5_rerank; W5_DATA=/workspace/koz3/w5data W5_ROOT=/workspace/koz3/wt/w5 python /workspace/koz3/orch/bot-W4/scripts/v3a_levers.py --levers <L> [--pl rp]
