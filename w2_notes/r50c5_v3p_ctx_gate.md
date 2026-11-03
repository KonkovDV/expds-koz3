# r50c5 → V3P gate in prod context (01.11.2026 22:15 MSK): FAIL
- V3P reproduced: 0.3337867244 (|d| 4e-11). V3P_ctx (v3_on output_floors) = 0.345852.
- B = V3P_ctx + li_r50c5: 0.348243, Δ +0.002391, P 0.6565; floors povr +0.0216 / tselye −0.0077 / gor −0.0151 / fort −0.0052; new34 −0.0132, nolidar −0.0019 → **FAIL**.
- B against V3P without context: +0.0145, P 0.99. This is report-only; the prod context explains the gap.
- Details and how to rerun: w2_gate/r50c5_v3p/README.md and gate_r50c5_ctx.json. Archive fit was not assessed, because that step only runs on PASS.
