# W4 gate C1FP (C1F-all + FP1) on v4 → candidate v5 — **FAIL** (02.11 MSK)

Protocol: honest-10 over v3_on (b447649) base, freeze rule (only sp_dfine τ retuned for touched classes on train regions; all other groups = taus_V3P),
centre = C1 per-fold nested (d0,s) from v4 prep_summary (s=0.75, d0∈{0,0.25}) on all 5 sp_dfine classes × FP1 frame (d0=0.5, edge=0.5) on FPC (kp,kt,arkh,fort);
production replay via `[post.w4_sat]` (center+frame+thresholds_sp_dfine); paired site-family bootstrap 2000, seed 0. Bar: Δall≥+0.003, P≥0.95, new34≥0, nolidar≥0, floors (kp/kt/gor −0.002, others −0.005).

| vs | Δall | P | p5 | new34 | nolidar | floor fails |
|---|---|---|---|---|---|---|
| base v3_on (0.345439) | **+0.01344** | 1.000 | +0.00663 | +0.00194 (P0.845) | +0.02117 | **kurgany_tselye −0.00227** |
| v4 = C1 (0.361641) | **−0.00276** | 0.037 | −0.00605 | +0.00119 | **−0.01188** | kp −0.00226, kt −0.00402, arkh −0.01277 |

C1FP all = 0.358883 (< v4 0.361641). Gate-side nested: C1FP 0.34647 vs C1 0.34895 (base 0.33379) — consistent.
Verdict: **FAIL / do not enable**. v5 not formed; v4 (C1F-all) remains the candidate. FP1 frame on top of all-class centre over-suppresses (double prior on kp/kt/arkh).

Out: `orch/bot-W4/comp/c1fp/out/out_C1FP.geojson` sha256 `d844037543fb6e977af2fe2fc39876536f581ed144c55de4d42cfcddad15d63e`.
Code: `orch/bot-W4/comp/c1fp/{c1fp_prep,c1fp_replay,c1fp_score}.py`, `run.sh` (sidecar of v4build/gate/c1fx_*; originals untouched). JSON: `runs/gate_C1FP_v5.json`.
No LB, no push, no prod flags, CPU only.
