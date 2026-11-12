> **12.11.2026.** Срез 01.11. Лучший балл тогда был 0,41238. Сейчас 0,43274 (`docs/LB_LOG.md`). Карта документов — `docs/DOCS.md`.

# KOZ No.3 — Handoff (2026-11-01)
Goal: archaeology object detection (KOZ No.3). Deadline **11.11.2026**. Target LB ~0.61; our best LB **0.41238** (v2, commit eb268d6).
Reports index: [snapshot/2026-11-01/INDEX.md](../snapshot/2026-11-01/INDEX.md) · PR https://github.com/KonkovDV/expds-koz3/pull/28

## Gates (honest-10 vs B′ 0.3158)
| Lever | Status | Result |
|---|---|---|
| sat2 REPLACE scale1.0 | PASS | 0.326447 vs A 0.314615 (+0.0118, P1.0, damaged not worse) |
| FR1h | soft PASS | vs B′ +0.0029, P0.999 — awaiting Dmitrij's "да" to submit |
| li-post full | FAIL | −0.005 |
| geo L3 hull | FAIL | 0 |
| sat-recall | FAIL | −0.0015 |
| pixgbm-cr | FAIL | equivalence fail, maggr −0.017 (reports/bot6_pixgbm.md @c282f682) |
| r50c5, WSR, ring, peak-rerank, dino-gor, sp_large, sat 8m cluster, ortho multiclass | CLOSED | do not reopen |

## Running
- Combined v2+sat2+FR1h gate + pack build (orchestrator, in progress)
- large-gate v2: elenakonkova/koz3-w4-large-gate (re-pushed after fix ee038449; earlier Δ0 was missing-ae-OOF bug)
- ae-dfine-gate: dmitrijkv/koz3-w1-ae-dfine-gate
- lidar-aug f2 (dmitrijkv), f0 (elenakonkova)

## Pending
- Dmitrij's "да" for FR1h / combo submission (Elena submits)
- Rotate S3 keys after lidar folds finish
- Retry koz3-w3-geo5-feat-base mirror to CPU accounts

## Update 01.11 ~21:45 MSK (takeover bot) — see reports/h2h_20261101/RESULTS.md
- large-gate FAIL (fixc_nested Δ−0.00012 vs A); ae-dfine-gate FAIL (flag_dead smoke); lidar-aug f2 fold FAIL (0.343 vs ref 0.401, early-stop); lidar-aug f0 v5 (folds 3,4) still RUNNING.
- Same-engine h2h: v3_on 0.3338 > v2+sat2 0.3236 (Δ−0.0102); v3_on+sat2 +0.0019 P0.941 FAIL. Final = v3_on; FR1h dropped.
