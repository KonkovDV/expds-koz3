# PREREG h2h sat2 vs v3_on (handoff bot, 01.11.2026 ~21:45 MSK), written BEFORE measurement

Question: on ONE engine, which line is higher — v3_on (V3P, sp_dfine replaces sp_*) or v2+sat2 (sp_det2 REPLACE of sp_or kurgans) — and does adding sat2 on top of V3P help?

Engine: orch/bot-W4/work/v3_gate_lib.py machinery unchanged (d3_gate TableEngine on w5data/work/table.pkl, family_folds 10,
honest per-fold resweep of all groups, unseen=drop, paired bootstrap by family 2000 seed 0). Base B0 = v2clean (as PREREG_w4_v3on).
sat2 OOF: w1/sat2/pkg/data/sp_det2_oof_notta.geojson (threshold_group sp_det2, kurgany_* only).

Variants:
* V3P — reproduction, must equal 0.333787 (sanity).
* S2 — B0 minus rows with class kurgany_* and group in {sp_or, sp_focal, sp_focal2, sp_arch} (= stack_replace_sp.py rule), plus sat2 OOF.
* U (PRIMARY) — V3P plus sat2 OOF (sp_dfine and sp_det2 as separate jointly-resweept groups).

Decisions:
* Ranking (informational, no gate): S2 vs V3P on the same engine.
* U accepted into final only if U vs V3P: delta>=0.003, P>=0.95 (Bonferroni over 2 post-hoc variants S2,U), per-class floors
  (>=-0.002 kurgany/gorodishcha, >=-0.005 others), new34>=0, nolidar>=0. Otherwise final stays v3_on.
Known deviation: sat2 OOF folds (5) are not the honest-10 family folds; thresholds still nested by honest-10.
