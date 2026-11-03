# (c′) Or localization checklist (bot5)

1. `assert_dfine_aeor_or_nonzero.py` passed (n_raw/Or dets > 0) — else HARD FAIL.
2. Run this script on `out_on/result.geojson` + GT (+ dump_on if present).
3. Record: n_or, soft_tp@2m, median dist_any_m, within_2/50/200.
4. Compare to bot2 A3: median~388 m / 0 TP@2m → if similar, localization domain issue (not CRS); do not accept on count>0 alone.
5. PROTOCOL §5 still required: ΔALL>0 and no layer drop >0.01 vs flags-OFF Docker.
6. Append EXPERIMENTS.csv + update reports/bot5_v4.md.

Generated summary: `summary.json`
