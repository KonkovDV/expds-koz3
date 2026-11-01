# bot7 geo-gate — real honest-10 scoreboard (fix stub)

## B′ config
**B′ = 0.315800** = Ae-ON / flags-off family honest-10 (PLAN_STICKY / HANDOFF / tta-gate).
Measured historically: `stack_arch` (+Ae) → `w5_rerank/prep.py` → `gate_v6c`; pair artifact `bprime_counts.pkl`.
**A = 0.314615** = `base_counts.pkl` on `stack_arch` without Ae bump (|A−B′| ≈ 0.001185 < 0.002).

## Levers
| Lever | Flags | Mode |
|---|---|---|
| L3_geo_convex_hull | `post.geo_convex_hull=true` | **post-only** on stack (FIRST) |
| L1_extremum_snap | `geophys.maggr.finds_points.extremum_snap_m=3.0` | MagGr reinfer; `GEO_GATE_REINFER=1` or **SKIPPED** |
| L2_finds_nms | `…finds_nms_m=1.5` | MagGr reinfer; same |
| ALL | all three | without reinfer ≡ L3 only |

## Contract
- No silent null `DONE` stubs. Missing w5base/stack/gt/counts → raise.
- Baseline must be within ±0.002 of B′ (or vs-ref A/Bprime); `GEO_GATE_DRYRUN=1` skips check.
- Gate: Δ≥0.003, p_gt0≥0.9, damaged not worse.
- Package: `/workspace/kaggle/push/geo-gate/` slug `dmitrijkv/koz3-bot7-geo-gate`
- Datasets: `elenakonkova/koz3-w2-gate` + `dmitrijkv/koz3-gt-train-v2` + `dmitrijkv/koz3-s3-cred`
- Push (bot1): `orch/pool_kaggle.sh dmitrijkv kernels push -p /workspace/kaggle/push/geo-gate`

## Dry-run (box)
`GEO_GATE_SKIP_S3=1 GEO_GATE_DRYRUN=1 GEO_GATE_BOOT=50` → baseline/L3 scores **non-null** (0.0 on synthetic), L3 `n_hull=1`, L1/L2 SKIPPED.

## Expected Kaggle runtime (CPU)
- S3 200–217 already cached: ~0–5 min skip
- prep+gate × (baseline + L3 [+ ALL≡L3]): **~25–90 min** (boot=2000)
- L1/L2 reinfer: not enabled in this build (SKIPPED); would add MagGr classic hours if wired
