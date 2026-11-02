# W4 gate: dfine2 FULL (a+b, D-FINE folds 0..4, 132/132 regions) ENS8 on v4, vs v4/out_C1. Result: **FAIL** (02.11, ~16:59 MSK)

Both kernels are COMPLETE. `dmitrijkv/koz3-w2-dfine2-a` covers folds 0 and 3 (38189 dets, 18 regions). `dmitrijkv/koz3-w2-dfine2-b` covers folds 1, 2 and 4 (55065 + 78515 + 70927 = 204507 dets, 114 regions). Its log ends «ГОТОВО 204507 детекций; ошибки фолдов: []». `skipped_regions_no_fold` is [] in both reports.

Coverage: all **132** regions get a dfine2 member (242696 CL5 dets). No region keeps the ref-only sp_dfine.

**Protocol.** Same as `gate_dfine2a_vs_v4.md`:
- v4 = C1 production replay of b447649 with `[post.w4_sat]` centre. Honest-10 site-family folds; freeze rule (only sp_dfine τ is re-tuned, on train). Paired site-family bootstrap, 2000 resamples, seed 0.
- Member: every region's sp_dfine is replaced by PREREG ENS8, `cluster_circles(ref model0 + dfine2 model1, 4 m, merge 8 m, mean, n_models 2)` → `finalize(merge_within_markup)`. Group `sp_dfine` is kept (= ens8_mean_as_sp_dfine). This gives 107839 circles against 85069 ref.
- Bar: Δall ≥ +0.003, P ≥ 0.95, new34 ≥ 0, nolidar ≥ 0. Class ΔF1 floors: ≥ −0.002 for kp/kt/gor, ≥ −0.005 for arkh/fort.
- Sanity: identity circles are exact (85069 = 85069). ID reproduces C1's chosen (d0,s), τ and gate-nested score (0.348952). The replayed ID out is **byte-identical to out_C1** (sha 3d286d22…).

| variant vs v4 out_C1 (0.361641) | score | Δall | P | p5 | Δnew34 | Δnolidar | class ΔF1 | verdict |
|---|---|---|---|---|---|---|---|---|
| **E8C1** (decisive) | 0.366889 | +0.00525 | 0.9545 | +0.00013 | -0.00116 | +0.00978 | kp +0.00535, kt -0.00300, gor +0.04309, arkh -0.02209, fort -0.00020 | **FAIL** |
| E8C1f (report, C1 (d0,s) frozen) | 0.365211 | +0.00357 | 0.8705 | -0.00200 | -0.00116 | +0.00613 | kp +0.00134, kt -0.00198, gor +0.03484, arkh -0.01658, fort -0.00007 | **FAIL** |

**Why E8C1 fails.** Δall (+0.00525) and P (0.9545) both clear the bar. It fails on three other checks:
- new34 is −0.00117 (P 0.172).
- The kt floor fails: −0.00300 against −0.002.
- The arkh floor fails: −0.02209 against −0.005.

gor gains a lot (+0.0431) and nolidar improves (+0.0098, P 0.92).

For reference, the comparison against v3_on base (0.345439):
- E8C1: +0.02145, P 1.0
- E8C1f: +0.01977, P 1.0

Both also have new34 −0.00042 against base.

Gate-side nested scores: C1 0.348952, E8C1 0.354375, E8C1f 0.352822.

**Caveats.**
- This compares against v4 out_C1, not the PREREG `B_ens8_mean_ctx_vs_V3P_ctx` (`gate_dfine2_ctx.py`), so it is not the W2 PREREG decisive gate.
- The arkh regression is large and consistent with the a-only gate (−0.0078 on 18 regions). It suggests a class-specific ENS8 (e.g. excluding arkh from the dfine2 merge) could pass. That would be a new, post-hoc variant needing its own PREREG.
- The all-folds early stop was not run (optional).

**Code** in `comp/dfine2full/`:
- `dfine2full_prep.py`, `dfine2full_replay.py`, `dfine2full_score.py`, `run.sh`: sidecars of dfine2a. Only OUT/D2P paths changed. Originals are untouched.

**Outputs** in `comp/dfine2full/out/`:
- `out_{ID,E8C1,E8C1f}.geojson`
- `dfine2full_gate_vs_{C1,base}.json`
- `prep_summary.json`
- `../sha.txt`

JSON summary: `runs/gate_dfine2_full_vs_v4.json`.

No LB submission, no push, no production flags, CPU only. v4 stays primary.
