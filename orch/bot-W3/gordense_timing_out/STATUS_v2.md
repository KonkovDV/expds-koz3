# gordense-timing-k v2 — COMPLETE (VALID, timing-only, head 487c48e4)
1. COMPLETE ~11:52 MSK (start 11:37), kernel elapsed 919 s, T4, commit 86a60c5; outputs → gordense_timing_out/v2/.
2. Data OK: S3 175 tif / 132 UTM / 132 regions, 5.40 GB (Kaggle Secrets ConnectionError → S3 fallback worked).
3. ON wall 467.6 s vs OFF 278.7 s → Δ +188.9 s (gor_dense phase 180.9 s, raw 259, 132/132 regions).
4. gor_dense APPLIED: 0× UNAVAILABLE, 0 Traceback, rc=0 both; result md5 ON dc19857e ≠ OFF 63b7d87c.
5. Targets met: ON 467.6 s < 900 (bot10) < 960 (working) < 1800 (platform).
6. GPU util max 100%, mem max ~3.2 GiB (ON 3173 / OFF 3145 MiB).
7. Caveats: sp_dfine OFF (D-FINE SpOr not mounted); gor_v3 head 487c48e4 (not legacy acb95c52) → TIMING-only, not a quality gate.
8. Lease bot-W3-gordense (→14:06) releasable; elena GPU free. No f234/LB/main touched.
