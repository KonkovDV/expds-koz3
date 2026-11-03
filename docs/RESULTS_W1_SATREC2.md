# RESULTS W1-SATREC2 (02.11.2026, bot-W1-satrec)

PREREG docs/PREREG_W1_SATREC2.md (1ff6643, committed before measuring). Family m=6, Bonferroni P ≥ 0.98333.
Base v3_on PH replay (b447649) all = 0.345439 (sanity vs 0.345439: True); gate-side base 0.3337867.

## Production-replay gate (honest-10 nested)

| var | Δall | P (raw) | p5..p95 | Δnew34 | Δnolidar | P4 fails | verdict | gate-side nested Δ |
|---|---|---|---|---|---|---|---|---|
| A1 | +0.0135 | 0.9970 | +0.0049..+0.0225 | -0.0017 | +0.0231 | kurgany_tselye -0.0074 | FAIL (P4, new34>=0) | +0.01137 |
| A2 | +0.0023 | 0.6810 | -0.0062..+0.0095 | -0.0047 | -0.0097 | kurgany_povrezhdennye -0.0080, kurgany_tselye -0.0044 | FAIL (d_all>=0.003, P>=0.98333(bonf), P4, new34>=0, nolidar>=0) | -0.00072 |
| A3 | +0.0107 | 0.9900 | +0.0027..+0.0185 | -0.0023 | +0.0106 | kurgany_tselye -0.0036 | FAIL (P4, new34>=0) | +0.00793 |
| B1 | +0.0074 | 0.8660 | -0.0040..+0.0186 | -0.0049 | +0.0063 | kurgany_povrezhdennye -0.0125 | FAIL (P>=0.98333(bonf), P4, new34>=0) | +0.00471 |
| B2 | +0.0086 | 0.8845 | -0.0034..+0.0202 | -0.0023 | +0.0108 | kurgany_povrezhdennye -0.0118 | FAIL (P>=0.98333(bonf), P4, new34>=0) | +0.00609 |
| B3 | +0.0049 | 0.9060 | -0.0012..+0.0114 | -0.0002 | +0.0066 | kurgany_povrezhdennye -0.0031 | FAIL (P>=0.98333(bonf), P4, new34>=0) | +0.00248 |

Per-class ΔF1 (production):

- A1: kurgany_povrezhdennye +0.0095, kurgany_tselye -0.0074, gorodishcha +0.0583, arkhitektury +0.0331, object_poly +0.0000, fortifikatsii +0.0005, finds_points +0.0000
- A2: kurgany_povrezhdennye -0.0080, kurgany_tselye -0.0044, gorodishcha +0.0091, arkhitektury +0.0413, object_poly +0.0000, fortifikatsii +0.0006, finds_points +0.0000
- A3: kurgany_povrezhdennye +0.0018, kurgany_tselye -0.0036, gorodishcha +0.0368, arkhitektury +0.0530, object_poly +0.0000, fortifikatsii +0.0003, finds_points +0.0000
- B1: kurgany_povrezhdennye -0.0125, kurgany_tselye +0.0023, gorodishcha +0.0608, arkhitektury +0.0145, object_poly +0.0000, fortifikatsii +0.0022, finds_points +0.0000
- B2: kurgany_povrezhdennye -0.0118, kurgany_tselye -0.0018, gorodishcha +0.0671, arkhitektury +0.0233, object_poly +0.0000, fortifikatsii +0.0020, finds_points +0.0000
- B3: kurgany_povrezhdennye -0.0031, kurgany_tselye -0.0011, gorodishcha +0.0250, arkhitektury +0.0230, object_poly +0.0000, fortifikatsii +0.0007, finds_points +0.0000

## Chosen hyperparameters per fold (nested, training folds only)

- A1: d0 0.0 s_o 0.75 s_kt 0.5; d0 0.25 s_o 0.75 s_kt 0.5; d0 0.0 s_o 0.75 s_kt 0.75; d0 0.25 s_o 1.5 s_kt 0.5; d0 0.0 s_o 1.0 s_kt 0.5; d0 0.25 s_o 1.5 s_kt 0.5; d0 0.0 s_o 0.5 s_kt 0.5; d0 0.25 s_o 0.5 s_kt 0.5; d0 0.0 s_o 1.0 s_kt 0.25; d0 0.25 s_o 1.5 s_kt 0.75
- A2: d0 0.25 s_o 1.25 s_kt 0.0; d0 0.25 s_o 0.5 s_kt 0.0; d0 0.0 s_o 0.75 s_kt 0.0; d0 0.25 s_o 1.5 s_kt 0.0; d0 0.25 s_o 0.5 s_kt 0.0; d0 0.25 s_o 0.75 s_kt 0.0; d0 0.25 s_o 0.75 s_kt 0.25; d0 0.0 s_o 0.5 s_kt 0.0; d0 0.25 s_o 0.5 s_kt 0.25; d0 0.25 s_o 0.5 s_kt 0.0
- A3: d0 0.25 s_o 0.25 s_kt 1.0; d0 0.25 s_o 0.75 s_kt 0.5; d0 0.0 s_o 0.75 s_kt 1.0; d0 0.25 s_o 0.75 s_kt 1.0; d0 0.0 s_o 1.0 s_kt 1.0; d0 0.25 s_o 0.75 s_kt 0.5; d0 0.5 s_o 0.25 s_kt 0.75; d0 0.25 s_o 0.5 s_kt 1.0; d0 0.0 s_o 0.75 s_kt 0.25; d0 0.25 s_o 0.5 s_kt 0.75 (admissible per fold: [18, 26, 20, 20, 34, 46, 20, 17, 49, 71])
- B1: τ0.4 d0 0.0 s_o 1.25 s_kt 0.75; τ0.5 d0 0.25 s_o 0.75 s_kt 0.75; τ0.55 d0 0.0 s_o 0.75 s_kt 0.5; τ0.65 d0 0.5 s_o 0.75 s_kt 0.75; τ0.4 d0 0.0 s_o 1.25 s_kt 0.75; τ0.5 d0 0.25 s_o 0.75 s_kt 0.75; τ0.5 d0 0.25 s_o 0.75 s_kt 0.5; τ0.5 d0 0.25 s_o 0.75 s_kt 0.75; τ0.5 d0 0.25 s_o 0.75 s_kt 0.75; τ0.5 d0 0.25 s_o 0.75 s_kt 0.75
- B2: τ0.55 d0 0.25 s_o 1.25 s_kt 0.25; τ0.5 d0 0.25 s_o 0.75 s_kt 0.5; τ0.55 d0 0.0 s_o 0.75 s_kt 0.25; τ0.5 d0 0.25 s_o 0.75 s_kt 0.5; τ0.5 d0 0.25 s_o 0.75 s_kt 0.5; τ0.5 d0 0.25 s_o 0.75 s_kt 0.5; τ0.5 d0 0.25 s_o 0.75 s_kt 0.5; τ0.5 d0 0.25 s_o 0.75 s_kt 0.5; τ0.5 d0 0.25 s_o 0.75 s_kt 0.75; τ0.5 d0 0.25 s_o 0.75 s_kt 0.5
- B3: τ0.5 d0 0.0 s_o 0.5 s_kt 0.75; base; base; base; base; τ0.45 d0 0.5 s_o 1.5 s_kt 1.25; τ0.55 d0 0.0 s_o 0.5 s_kt 0.25; base; base; τ0.4 d0 0.0 s_o 0.75 s_kt 0.75 (admissible per fold: [28, 1, 1, 1, 1, 2, 39, 1, 1, 25])

## Verdict

All 6 FAIL. No variant has Δnew34 ≥ 0, and every variant breaks a kurgan P4 floor (kt for A1/A3, kp for B1/B2/B3, both for A2).
- A1 (C1+, per-class s, grid up to s 1.5 / d0 0.5): Δ +0.0135, P 0.997 (≥ 0.98333), nolidar +0.023. Fails new34 −0.0017 and kt −0.0074. The kt-specific strength picked on training folds (mostly s_kt 0.5) does not carry over to the held folds.
- A2 (modality rule, prior only in lidar-free regions): the gain disappears (Δ +0.0023, P 0.68), and new34 and nolidar both get worse. The per-class τ of sp_dfine are fit over all regions, so dampening only part of the regions moves τ everywhere.
- A3 (constrained: training P4 floors plus Δ on H ≥ 0): Δ +0.0107, P 0.990, new34 −0.0023, kt −0.0036. The constraints hold on the training folds but fail on the held folds.
- B1/B2 (single τ plus prior): Δ +0.007..0.009, P < 0.9, kp −0.012.
- B3: Δ +0.0049, P 0.906, new34 −0.0002, kp −0.0031. The constraint fell back to base in 6 of 10 folds.
Conclusion: the centre-prior recall lever on D-FINE does not carry over to new34 or kurgan classes in any of the six forms. Flags stay off; nothing is proposed for v3.

## Artifacts
- expds/sat_recall.py: sat_center_prior adds s>1, s_by_class, only_without_lidar, nl_group (legacy d0/edge path unchanged; flag off). tests/test_sat_recall.py: 3 pass.
- tools/satrec2/: nl_regions.py (modality rule list), s1b_prep.py (gate-side nested; per-class separable combos), s2b_replay.py (PH replay with the new prior spec on the finish input), s3b_score.py, run_replays.sh.
- docs/results_w1_satrec2/: s3b_gate.json, prep2_summary.json, nl_regions.json, H_regions.json.
- Runs: box, nice 10, ≤4 threads, CPU; s1b ~11 min, replays ~15 s/variant, scoring 15 s.
