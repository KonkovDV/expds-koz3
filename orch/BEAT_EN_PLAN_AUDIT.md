# BEAT_EN_PLAN_AUDIT — audit of BEAT_EN_WORLD_PLAN_20261101 (holes, risk to 0.41238, DOC-1)

A-executor, 01.11 14:27 MSK (UTC+3). Read with `BEAT_EN_WORLD_PLAN_20261101.md`, `HYPER_DEEP_20261101_1410.md`, `out/sat2-f0/REPLACE_HONEST.md`.

## A. Risk to 0.41238
The bot never submits. Every LB step is an Elena upload, and the archive zip (sha 8f3f2671…) stays the rollback.
The real risk is the **final selection**, not the probes:
- If EN-line PIXGBM is ruled out (DOC-1), the final falls to ≈ **0.378**.
- The clean zip is the only DOC-1-safe pack with PIXGBM-level geo. It has check_submission PASS and Docker smoke PASS, but **no LB number yet**.

## B. Holes found (severity H/M/L) and fixes
| # | Sev | Hole | Fix / owner |
|---|---|---|---|
| 1 | **H** | **S3 path(b)**: B pushed R-augx with AWS keys inlined at push time and scrubbed only locally. By path(b) design (`_materialize_s3_env`), the **pushed Kaggle kernel version very likely keeps the keys** in a private kernel's version history. I did not verify this, to avoid handling the secret. | After v2 ends: rotate the S3 keys, or push a scrubbed version and delete the old one. **Needs Elena «да»** (rotation and deletes are outside the box). Never reuse path(b) for folds 1–4; use Kaggle Secrets. |
| 2 | **H** | Clean zip = **FR1h overlay + clean PIXGBM** in one pack. One LB number cannot separate the clean-geo effect (vs EN pix) from FR1h (+0.004 OOF). | Accept as the insurance probe. If LB < 0.41238 − 0.003, the next probe is archive + clean-pix without FR1h (build on CPU, ~5 min). |
| 3 | **H** | det2 inference cost on the platform is **unmeasured**: ConvNeXt-T CenterNet with d8 TTA on CPU over all sat tiles, inside 1500 s with pix 900 s. If it does not fit, the plan's step 2 is worthless. | CPU timing of det2 on one test-size sat region **before** spending GPU on folds 1–4 (cheap, box). Fallback: no TTA / d2. |
| 4 | M | Sat fold0 = **1 family**. Today's fold0 "PASS/FAIL" calls (sat2 gate, sp_large) rest on one family, which has high variance. | Folds 1–4 are required before any keep/kill that matters. Fold0 only screens. |
| 5 | M | Replace with partial OOF contaminates the shared tau (optical −0.0048 today). A naive "replace sp_focal" with fold0-only would look like a FAIL even for a good model. | Protocol rule: a replace gate needs all-fold OOF and its own group tau (written into the plan §4). |
| 6 | M | box honest-10 base A (0.31462, stack_arch) ≠ scoreboard honest (Elena PC work dir, 0.3295). The two are not comparable. | Always report Δ vs the same base. scoreboard runs need the F:/expds3/work dir (not on box). |
| 7 | M | Pack `budget_s=900` is counted from process start. The Kaggle core speed is used as a proxy for the platform. A lleaves→LightGBM fallback (sha mismatch / ISA) costs ~1100 s → classic → −35 % geo preds (the 0.378 mechanism). | x86-64-v2 target chosen for safety. The checker smoke PASS imports lleaves. Keep the 900 headroom. Watch the LB log for "classic". |
| 8 | M | stack6 archive filters (multipole_object_poly, filter_finds_points) are OFF in the clean pack and ON in the archive. This geo delta is unmeasured on LB. | E1_G1 was gated OFF. Leave OFF; the flag is Elena's. |
| 9 | L | In-sample geo gate (P .79 / R .67) is optimistic. The honest figure is OOF freeze .6925/.6738. | Quote OOF only. |
| 10 | L | EN-gap estimates (+0.053 / +0.051) come from the gap model, not measurements. The plan ceiling ≈0.46 is a hypothesis. | Keep ranges and gates. |
| 11 | L | Disk 14 GB free. Folds 1–4 pulls (sat2 ≈ 10 GB/fold-set) will not fit. | Elena «да» to delete STOP artefacts (dino-gor 6.8 G, sp_large-f0 4.6 G, wsr) before any fold pulls. Pull only geojson/OOF, not .pt. |

## C. DOC-1 (EN) reminder
- 0.41238 = archive eb268d6 carries the **EN-line `geo_pixgbm.py` + models/pixgbm**. If the organisers rule EN-derived code inadmissible, the final drops ≈ −0.034.
- The clean zip passes the DOC-1 guard (`check_archive_clean.py` OK; EN geo_pixgbm.py removed by name; clean pix from wt/geo-clean 552c5bc / 579a92f, local, not pushed).
- PR #20 (EN retrain) = DOC-1 risk, do not merge. Today's world/EN research = docs/ledger only, no code or weights.
- **Decision for Elena:** which zip is "final": the archive (EN-line) or the clean (DOC-1-safe)? Recommendation: probe the clean zip non-final first; if LB ≥ 0.409, make clean the final.

## D. Verdict
The plan is sound in order (insurance → LiDAR in flight → sat replace with all-fold OOF → heterogeneity). It still has three blocking holes before any GPU «да»:
**#1 key rotation**, **#3 det2 CPU timing**, **#11 disk**. P(> EN 0.505) is low. P(> 0.41238 with the clean final) is high, given the FR1h OOF and geo parity.
