# EXPERIMENTS — sat2 gate 01.11.2026 (bot-W1-sat / bot2)

| # | Изменение | Δ | P(p_gt0) | Статус | Отчёт |
|---|---|---|---|---|---|
| S2-R | sat2 REPLACE soft-PASS (Drive) | ΔB′ **+0.00472** (0.320516) | **0.838** (soft bar ≥0.8) | 🟡 soft-PASS | HANDOFF_NOW_20261101 |
| S2-G | sat2 REPLACE honest-10 full (boot=4000) + sat_thr_relax scale sweep | TBD after kernel | TBD (≥0.9 target) | ⏳ package ready — bot1 push `elenakonkova/koz3-w1-sat2-gate` | `bots/w1-sat2-gate/REPORT_sat2_gate_20261101.md` |

Notes:
- P = paired bootstrap p_gt0, not precision.
- Package: `/workspace/kaggle/push/sat2-gate/` manifest `d6323ebb…`; OOF `d28feee1…`.
- expected ΔLB if pass_full: +0.003…+0.006; hold LB 0.41238.
- default-OFF: `[sat_thr_relax] enabled=false`.
