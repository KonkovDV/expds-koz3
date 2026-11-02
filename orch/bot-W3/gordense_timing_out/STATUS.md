# gordense-timing-k STATUS (2026-10-31 11:24 MSK, bot D)
1. elenakonkova/koz3-w3-gordense-timing-k COMPLETE (elapsed 239 s, T4); acc1 GPU free now. Outputs here.
2. TIMING INVALID: xfer gave 4 tif/4 UTM (0.009 GB) vs expected 175/132 (5.4 GB) — only 4 regions processed.
3. gor_dense NOT applied in ON run: GOR_DENSE_UNAVAILABLE, gor_gate md5 mismatch (gor_v3_head.npz is 487c48e4…, config expects acb95c52…).
4. ON wall 18.1 s / OFF 14.2 s, Δ+3.9 s; ON==OFF result md5 f3f58e1b… (identical, 4 raw optics) — no gor_dense signal.
5. sp_dfine OFF (f0 not mounted). GPU util max 3–8%.
6. Fix before rerun: correct gor_v3_head.npz (or md5 in config) in DS + full xfer shard download (s000–s009 found but only 4 tif pulled).
7. dmitrijkv/r50c5-f234 untouched. No LB. Lease bot-W3-gordense (→13:04) can be released.
# gordense-timing-k v2 — COMPLETE (VALID, timing-only, head 487c48e4)
1. COMPLETE ~11:52 MSK (start 11:37), kernel elapsed 919 s, T4, commit 86a60c5; outputs → gordense_timing_out/v2/.
2. Data OK: S3 175 tif / 132 UTM / 132 regions, 5.40 GB (Kaggle Secrets ConnectionError → S3 fallback worked).
3. ON wall 467.6 s vs OFF 278.7 s → Δ +188.9 s (gor_dense phase 180.9 s, raw 259, 132/132 regions).
4. gor_dense APPLIED: 0× UNAVAILABLE, 0 Traceback, rc=0 both; result md5 ON dc19857e ≠ OFF 63b7d87c.
5. Targets met: ON 467.6 s < 900 (bot10) < 960 (working) < 1800 (platform).
6. GPU util max 100%, mem max ~3.2 GiB (ON 3173 / OFF 3145 MiB).
7. Caveats: sp_dfine OFF (D-FINE SpOr not mounted); gor_v3 head 487c48e4 (not legacy acb95c52) → TIMING-only, not a quality gate.
8. Lease bot-W3-gordense (→14:06) releasable; elena GPU free. No f234/LB/main touched.
