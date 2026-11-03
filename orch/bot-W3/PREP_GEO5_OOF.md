# PREP geo5 OOF (bot D) — 30.10 ~23:05 MSK
A: MagGr FREEZE at E1_G1 (`w4geo/work/cstar/C_CetO3_Sr8G_noTerFl_L_E1_G1{,_ht}.geojson`). After seeds-b COMPLETE+pull → geo5 OOF hold lifted; may start (no LB).

## Gate
OOF geo5 folds: F1={200} F2={201..207} F3={208,209,210,215} F4={211,212,216} F5={213,214,217}; 18 sites; gt_train_v2; honest thresholds. See `orch/PIXGBM_RECOVER_20261030.md` §4–5.

## D role (§5)
CPU slots after seeds-b → OOF geo5 (5 folds × finds/poly). Scripts: `wt/w4-pix/scripts/pix_clean_{train,oof,eval}.py`. Data canon: `/workspace/koz3/w4geo/train/`. Work: `w4geo/work/`. venv: `w4geo/venv`.

## Sequence
1. seeds-b COMPLETE both usertest+aaaadg → pull `orch/bot-W3/seeds_b_out/` → release cpu-2.
2. Acquire free CPU (lease via A / pool) for geo5 OOF folds — prefer freed usertest/aaaadg cpu-2 or box ≤3 procs nice 10 ≤8GB.
3. No LB, no main push. Report A ≤10 lines on start + done.

## Blockers to check at start
- feat cache / models for n40r8 line under `w4geo/work/`
- whether A wants D to re-run full OOF or only fold package for frozen E1_G1 inference taus
