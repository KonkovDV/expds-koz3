# KDUAL-damaged-only PREREG — W3, 02.11 MSK. Committed before measurement. A new hypothesis after KDUAL FAIL (dae47a7).
Identical to prereg/KURGAN_DUAL_PREREG.md (base V3P 0.333787, li pairs, nested τ_dual per outer fold on train only, the same grid, the same criterion and ties), with one change:
- an extra copy is emitted only when the second class is kurgany_povrezhdennye (damaged), i.e. the first is tselye.
- Copies whose second class is tselye are never added.
Flag kurgan_dual_dmg, default false. Clean-room, no EN code.
Family: {KDUAL, KDUAL-dmg} = 2 tests → Bonferroni P ≥ 1 − 0.1/2 = 0.95.
Accept if all hold:
- Δ_all ≥ +0.003 and P ≥ 0.95;
- new34 Δ ≥ 0;
- ΔF1 damaged ≥ 0, ΔF1 intact ≥ 0, ΔF1 fort ≥ 0;
- nolidar Δ ≥ 0;
- W4 class floors hold.
