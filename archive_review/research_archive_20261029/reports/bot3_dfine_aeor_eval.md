# bot-3 — D-FINE Ae-ft → `[dfine_aeor]` gate eval (Docker) — 29.10.2026 ~20:00 MSK

| Field | Value |
|---|---|
| **Goal** | Analyze `koz3-dfine-ae-ft` out; PROTOCOL section 10 Docker e2e mock + honest-10 with `[dfine_aeor]` Ae-only ON vs OFF (`pantum/n7q:20261024`); PC queue on **C:** (F: full). |
| **Branch** | `bots/bot3-dfine-aeor-eval` @ **`ee8af7f918aed56ab5cb4371bc9618957d7385ac`** |
| **Kernel** | COMPLETE `dmitrijkv/koz3-dfine-ae-ft` — out `/workspace/kaggle/out/dfine-ae-ft`, ~2.9 GPU-h, `fold_errors=[]` |
| **LB** | **No.** `[dfine_aeor] enabled=false` in `configs/default.toml`. |
| **Runner** | `pc/runners/run_bot3_dfine_aeor_ae_ft_docker.ps1 -Variant final\|f4` |
| **Launch** | `launch_bot3_dfine_aeor_ae_ft_schtasks.ps1` / `…_f4_schtasks.ps1` |

Related: `reports/bot3_dfine_ae_kernel.md`, PROTOCOL section 10, bot5 Or assert.

---

## 1. Ae-ft artifacts

| Artifact | Fact |
|---|---|
| `STATUS.txt` | `DONE` mode=full mod=Ae iters=2000 final_iters=1500 n_det=2741129 total_sec=10478 |
| Final `dfine_ae/model.safetensors` | sha256 **`3e61987893b5da499482dc1af8520b0793894f73149fa3f2cddab8d1c9dfacad`** |
| Fold4 `dfine_f4/model.safetensors` | sha256 **`249b86fb9713cc1cb234cfbf11d239414092ef1af8178f434df2a38edfa94efa`** (2000/2000 iters) |

**Final ≠ FT fold3:** packaged final = old input `koz3-dfine-ae/dfine_f3` + **1** FT iter (not local `OUT/dfine_f3` sha `23cc5220…`). Prefer fold4 for packaging quality; gate still runs both.

### OOF F2 (thresholds)

| Class | tau | F2 |
|---|---|---|
| kurgany_tselye | 0.45 | 0.028 |
| kurgany_povrezhdennye | 0.5 | 0.173 |
| fortifikatsii | 0.35 | 0.184 |
| gorodishcha / arkhitektury | 1.01 OFF | — |

### `[dfine_aeor]` layout match

HF `config.json` + `model.safetensors`, id2label 5 classes, `thresholds_oof.json` keys `dfine_ae` / `ae_dfine`. Runner sets `markups=["ae"]`, `weights.ae=["models/dfine_ae"]`. default.toml untouched.

---

## 2. PC paths (C: JobDir / Wt / weights — F: inputs only)

### Variant `final` (sha prefix `3e619878`)

| Role | Path |
|---|---|
| JobDir | `C:\Sloi\pc_jobs\bot3_dfine_aeor_ae_ft` |
| Wt | `C:\Sloi\pc_jobs\wt_bot3_dfine_aeor_ae_ft` |
| Weights | `C:\Sloi\pc_jobs\weights\dfine_ae_ft\` |
| Box source | `/workspace/kaggle/out/dfine-ae-ft/dfine_ae/` + `models/dfine_aeor/thresholds_oof.json` |
| Launch | `pc/runners/launch_bot3_dfine_aeor_ae_ft_schtasks.ps1` (task `koz3_bot3_dfine_aeor_ae_ft`) |
| Lease | `leases/pc_bot3_dfine_aeor_ae_ft.lease` |

### Variant `f4` (sha prefix `249b86fb`)

| Role | Path |
|---|---|
| JobDir | `C:\Sloi\pc_jobs\bot3_dfine_aeor_ae_ft_f4` |
| Wt | `C:\Sloi\pc_jobs\wt_bot3_dfine_aeor_ae_ft_f4` |
| Weights | `C:\Sloi\pc_jobs\weights\dfine_ae_ft_f4\` |
| Box pack | `/workspace/kaggle/out/dfine-ae-ft-f4pack/` (flat + `dfine_ae/` + `models/dfine_ae/` + `models/dfine_aeor/thresholds_oof.json`) |
| Launch | `pc/runners/launch_bot3_dfine_aeor_ae_ft_f4_schtasks.ps1` (task `koz3_bot3_dfine_aeor_ae_ft_f4`) |
| Lease | `leases/pc_bot3_dfine_aeor_ae_ft_f4.lease` |

**bot-1 copy:** pack root → `C:\Sloi\pc_jobs\weights\dfine_ae_ft_f4\` so that `config.json`, `model.safetensors`, `thresholds_oof.json` sit at that root (runner also accepts nested `dfine_ae\` / `models\dfine_aeor\`).

Mock regions/GT remain on F: (read-only).

---

## 3. Runner rules (coordinator fixes)

- **ASCII-only** `.ps1` (verified: 0 non-ASCII bytes, no BOM).
- **No** `python -c` with double quotes — helper `pc/runners/bot3_ae_weights_preflight.py`.
- **Preflight:** size > 0 AND sha prefix match; host may header-validate; Docker container runs `--full-load` (safetensors full tensor load) — hard FAIL otherwise.
- **Launch:** schtasks wrappers (same style as bot4 lo-ablate).
- **Assert:** `scripts/assert_dfine_aeor_ae_nonzero.py` — FAIL if n_raw/Ae dets == 0.

---

## 4. PASS criteria

1. STATUS=DONE, `ENV=docker:pantum/n7q:20261024`
2. docker_preflight PASS (full load)
3. `assert_ae_nonzero` PASS (n_raw > 0)
4. `scores_mock.json` with ae delta ON−OFF
5. honest-10 if regions present (no other-layer drop > 0.01)

---

## 5. Risks

- Final 1-iter from old input f3 — weak vs FT f4; both queued.
- tselye F2 0.028 → possible zero ae recall delta even with n_raw>0.
- fold1 essentially untrained (1 iter).
- Host Windows venv forbidden for gate numbers (PROTOCOL section 10).
