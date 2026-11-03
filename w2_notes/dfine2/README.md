# P-W2-13 D-FINE-2: launch notes (02.11.2026, W2 / bot 3)

PREREG: `orch_bot-W2/PREREG_W2_cnx.md` § P-W2-13. Commit 1c0ddc0 (13:44 MSK), amendment 1 34f97d3 (13:48 MSK, data naming only). Both are before the push.

## Kernels (dmitrijkv, 2×T4, private, no S3, no secrets)
| kernel | folds | version | pushed | ETA COMPLETE | hard cap (DEADLINE_H) | lease |
|---|---|---|---|---|---|---|
| dmitrijkv/koz3-w2-dfine2-a | 0,3 | v1 | 02.11 13:49 MSK | ~15:45 MSK | 4.0 h → ≤17:50 MSK | slot:acc2/gpu-1 + kernel |
| dmitrijkv/koz3-w2-dfine2-b | 4,2,1 | v1 | 02.11 13:49 MSK | ~16:55 MSK | 5.0 h → ≤18:50 MSK | slot:acc2/gpu-2 + kernel |

Both were RUNNING at 13:50 MSK. Quota before launch: 11.01 h used, 18.99 h left of 30 (refresh 07.11 03:00 MSK). Expected spend is about 5 GPU-h.
QUEUE.json dispatch entries are `w2-dfine2-a` and `w2-dfine2-b` (status running). The watcher renews the leases, pulls output to `pull_to` on COMPLETE/ERROR, and releases the leases.

- Sources: `w2_kernels/koz3-w2-dfine2/make_dfine2.py` builds them from bot-9 `kaggle/2026-10-29/dfine_spor/full/koz3-dfine-spor-full.py` using asserted replacements.
  Flags: DFINE_NUM_DN=0, DFINE_SAMPLING=site (DFINE_SITE_POW 0.5), DFINE_QAUG2=1, DFINE_SEED_BASE=200, DFINE_GROUP=sp_dfine2. With the flags unset, behaviour is the same as bot-9.
- Data: private datasets `dmitrijkv/koz3-w2-dfine2-raw-a` (88 tif + gt_train_v2.geojson + manifest_a.json) and `-raw-b` (87 tif + manifest_b.json).
  Kaggle strips non-ASCII characters from file names, so the manifests map `<NNN>__[b]<kk>_SpOr.tif` back to the region and original file name.
  Tools: `w2_kernels/koz3-w2-dfine2/data_tools/{fetch.sh,ascii_rename.py}`.
- Local CPU smoke runs (both halves of the data, 4 iterations) finished EXIT 0: DN off (denoising_class_embed unused), 132 GT regions, OOF props group sp_dfine2 with the original tif name.

## Output layout (pulled by the watcher)
`/workspace/koz3/w2/oof/dfine2/dmitrijkv_koz3-w2-dfine2-{a,b}/`:
- `oof_dfine_f{k}.geojson`: EPSG:3857 ellipses, one per fold.
- `oof_dfine_spor.geojson`: all folds of the kernel.
- `dfine_report.json` and the kernel log.

No weights are saved. A PASS needs a weights run for the archive.

## Early stop: run as soon as a is COMPLETE (b is still running)
```
cd /workspace/koz3/wt/w2 && nice -n 10 /workspace/koz3/w4venv/bin/python w2_tools/early_stop_dfine2.py \
  /workspace/koz3/w2/oof/dfine2/dmitrijkv_koz3-w2-dfine2-a --folds 0,3 --out w2_notes/dfine2/early_stop_a.json
```
- Scope: f0+f3, 18 regions, 249 GT.
- Baselines from before the measurement: S(A) = 0.249916 (sp_dfine bot-9), S(C) = 0.251746 (A merged with a copy of itself, 8 m mean).
  The identity self-test gives FAIL on R2 (`early_stop_selftest_identity.json`).
- **FAIL if S(B) < S(A) − 0.02 (R1) or S(M) ≤ S(C) (R2).** On FAIL:
  - The Kaggle CLI has no cancel. Stop b in the Kaggle UI (Cancel run), or let it run out (at most the 5 h cap). Either way the decision is the same, and the b OOF is not used.
  - `orch/lease.sh release slot:acc2/gpu-2 bot-W2-lidar` and `orch/lease.sh release kernel:dmitrijkv/koz3-w2-dfine2-b bot-W2-lidar`.
  - `orch/enqueue.py set w2-dfine2-b status=failed`.
  - Record the result in PREREG and STATUS. The line is closed.
- On PASS, wait for b.

## Gate: after a and b are COMPLETE and the early stop is PASS
```
cd /workspace/koz3/wt/w2/w2_gate/dfine2_v3p && nice -n 10 /workspace/koz3/w4venv/bin/python gate_dfine2_ctx.py 2000 \
  /workspace/koz3/wt/w2/w2_notes/dfine2/gate_dfine2_ctx.json
```
- Runtime is about 2.5 min with BOOT 0, plus the bootstrap.
- The default D2 dirs are the pull_to paths above. The config is `v3on_src/configs/default.toml` next to the script, or the box copy at /workspace/koz3/w2/r50c5_gate/v3on_src (not committed).
- Self-checks:
  - S0: V3P = 0.3337867244.
  - S1: this script's circle path on ref alone = V3P.
  - V3P_ctx = 0.3458516904.
  - OOF covers all 132 regions with folds = ref.
- Dry run with identity D2 (BOOT 0, `gate_dry_identity_boot0.json`): S0/S1 OK, V3P_ctx reproduced, decisive Δ −0.0005, floors match v3on_floors.json.
- Decisive variant: `B_ens8_mean_ctx_vs_V3P_ctx`. PASS needs all of:
  - Δ ≥ +0.003 and P ≥ 0.95.
  - new34 ≥ 0 and nolidar ≥ 0.
  - Class ΔF1 ≥ −0.002 for kp/kt/gor and ≥ −0.005 for arkh/fort.
- Report-only variants: ens8_max, ens4_mean, ens8_mean_as_sp_dfine, and the variant vs V3P.
