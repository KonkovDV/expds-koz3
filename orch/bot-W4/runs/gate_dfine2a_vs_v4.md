# W4 gate: dfine2-a (P-W2-13, D-FINE folds 0+3) on v4, vs v4/out_C1. Result: **FAIL** (02.11, ~16:22 MSK)

Kernel `dmitrijkv/koz3-w2-dfine2-a` was COMPLETE. Its output was pulled with `orch/pool_kaggle.sh` into `/workspace/koz3/w2/oof/dfine2/dmitrijkv_koz3-w2-dfine2-a/`:
- `oof_dfine_f0.geojson`: 5494 dets, 2 regions
- `oof_dfine_f3.geojson`: 32695 dets, 16 regions
- `oof_dfine_spor.geojson`, `dfine_report.json`, the kernel log, and `dfine_f{0,3}/model.safetensors`. The weights are present even though the README says they are not saved.

**Protocol.** v4 = C1 production replay of b447649 with `[post.w4_sat]` centre. Honest-10 site-family folds; freeze rule (only sp_dfine τ is re-tuned, on train regions). Paired site-family bootstrap, 2000 resamples, seed 0.
- Member: on the 18 a-regions, sp_dfine is replaced by the PREREG ENS8 member: `cluster_circles(ref model0 + dfine2 model1, 4 m, merge 8 m, mean, n_models 2)` → `finalize(merge_within_markup)`. It keeps group `sp_dfine`, which is the PREREG report-only variant ens8_mean_as_sp_dfine adapted to v4. The other 114 regions keep the ref sp_dfine, because b is still running.
- Bar (PREREG P-W2-13, same as W4 C1FP): Δall ≥ +0.003, P ≥ 0.95, new34 ≥ 0, nolidar ≥ 0. Class ΔF1 floors: ≥ −0.002 for kp/kt/gor, ≥ −0.005 for arkh/fort.
- Sanity: the identity variant (ID: ref-only circles through the same code) reproduces C1's chosen (d0,s), τ and gate-nested score. Its replayed out is **byte-identical to out_C1** (sha 3d286d22…).

| variant vs v4 out_C1 (0.361641) | score | Δall | P | p5 | new34 | nolidar | floor deltas | verdict |
|---|---|---|---|---|---|---|---|---|
| **E8C1** (decisive: full C1 recipe re-run nested) | 0.362052 | +0.00041 | 0.560 | −0.00181 | +0.00111 | +0.00288 | **arkh −0.00775** (fail); kp +0.0005, kt +0.0018, gor +0.0044, fort +0.0001 | **FAIL** |
| E8C1f (report: C1 (d0,s) frozen, τ re-swept) | 0.363605 | +0.00196 | 0.886 | −0.00063 | +0.00111 | +0.00690 | **arkh −0.00689** (fail); kp +0.0011, kt +0.0036, gor +0.0106 | FAIL |

For reference, the comparison against v3_on base (0.345439):
- E8C1: +0.01661, P 1.0
- E8C1f: +0.01817, P 1.0

On v3_on base, v4 alone gives +0.01620.

Gate-side nested scores: C1 0.348952, E8C1 0.349376, E8C1f 0.350870.

W2's PREREG early stop (W4 copy of `early_stop_dfine2.py`, f0+f3, 18 regions, 249 GT) gives **PASS**. S(A) is 0.249916, which reproduces the preregistered value.

| S(A) | S(B) | S(M) | S(C) |
|---|---|---|---|
| 0.249916 | 0.308833 | 0.333706 | 0.251746 |

The member is strong standalone, but the a-only half does not clear the v4 bar: Δ is far below +0.003, P is below 0.95, and the arkh floor fails.

**Caveats.**
- This is an a-only partial gate: 18 of 132 regions changed.
- The PREREG decisive gate (`B_ens8_mean_ctx_vs_V3P_ctx`, `gate_dfine2_ctx.py`) needs all 132 regions, i.e. dfine2-b, and it compares against V3P_ctx. When b lands, re-gate the full ENS8 vs out_C1 with the same sidecar.

**Code** in `comp/dfine2a/`:
- `dfine2a_prep.py`, `dfine2a_replay.py`, `dfine2a_score.py`, `run.sh`: sidecars of c1fp. Originals are untouched.
- `early_stop_dfine2_w4copy.py`: paths patched only.

**Outputs** in `comp/dfine2a/out/`:
- `out_{ID,E8C1,E8C1f}.geojson`
- `dfine2a_gate_vs_{C1,base}.json`
- `early_stop_a.json`
- `prep_summary.json`

JSON summary: `runs/gate_dfine2a_vs_v4.json`.

No LB submission, no push, no production flags, CPU only. v4 stays primary.
