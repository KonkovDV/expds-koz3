# FOR BOT 1 — gor_dense timing on the PC (GTX 1650, CUDA). bot-5, 29.10.2026

Goal: wall time of the full main.py with gor_dense ON (v4 + PFnoD of bot 10), ORGATE OFF, on CUDA.
Targets: **working ≤ 960 s** (0.8·1200), **bot 10's criterion ≤ 15 min (900 s)**, platform limit 30 min (1800 s).
The platform is assumed to be an RTX 4090 (NOT measured); the GTX 1650 and the T4 (kernel bot-5/kaggle/gordense_timing/) are upper bounds.
For reference: CPU on the box, 2 threads, gor_dense phase alone = 825 s for 132 SpOr regions (work/gd/time_phase.log:133);
the public 48 regions on this same PC took 337 s (old version, bot-3/PLAN.md:147).

## 1. Code
Branch `bot5-gor-dense-v4` @ **86a60c5** (accepted by bot 7; flag OFF = v4 byte for byte, work/gd/e2e/rerun4.log), or bot 7's merged v4
if gor_dense is already in it. transformers **5.21.0** (vendor inference/lib/vendor or env of the same version; `load_backbone` → `import_dfine`).

## 2. Enable block (configs/default.toml; the commented block is right above `[gor_dense]`)
```toml
[gor_dense]
enabled = true
backbone = "models/dinov2-small"
gate_head = "models/gor_gate/gor_v3_head.npz"
head = "models/gor_dense/gor_dense_head.npz"
head_md5 = "e35570e10d17edac32bea8d26fa0f5f4"
centre_s = 250.0

[thresholds_by_markup.gor_dense]
gorodishcha = 0.824066

[thresholds_by_markup.dfine_ae]
gorodishcha = 1.01
```
+ `gorodishcha = 1.01` in `[thresholds_by_markup.sp_dfine]`. If bot 7's `make_on.py` (bot-7/work6/on/make_on.py, sp_dfine R) already writes
that table, add the key TO THAT TABLE; a second header is a TOML error. `[thresholds_by_markup.dfine_or] gorodishcha = 1.01` is already active (bot-6), don't duplicate it.
ORGATE: `[orgate] enabled = false` (the v4 default). For the gor_dense block, replace the existing `[gor_dense] enabled = false` instead of adding a second table.
Check: `python -c "import tomllib;c=tomllib.load(open('configs/default.toml','rb'));print(c['gor_dense']['enabled'],c['thresholds_by_markup']['gor_dense'],c['thresholds_by_markup']['sp_dfine'].get('gorodishcha'),c['orgate']['enabled'])"`
→ `True {'gorodishcha': 0.824066} 1.01 False`.

## 3. Weights (paths relative to the repo root; md5 checked: head_md5 in config, the rest via md5sum)
| path | bytes | md5 |
|---|---|---|
| models/dinov2-small/config.json | 547 | 689053885e41f529baca946d85d9b8dd |
| models/dinov2-small/model.safetensors | 88 249 960 | ea251be6e151ba8609ad0bc9b2e13a78 |
| models/dinov2-small/preprocessor_config.json | 436 | e40d527806779cbf0c7fd6fca552eddb |
| models/gor_gate/gor_v3_head.npz | 407 736 | acb95c52956d712a16c3dd389237a691 |
| models/gor_dense/gor_dense_head.npz | 204 984 | e35570e10d17edac32bea8d26fa0f5f4 |
| models/dfine_f0/{config.json, model.safetensors} (sp_dfine R) | — | def4fb2d30fdad17ed43a02fb1ebfc2e / 3fe57318c9a17ccc0351fc00aae47f17 |

Sources on the box: dinov2 = bot-5/tmp/hf/hub/models--facebook--dinov2-small/snapshots/ed25f3a31f01632728cabb09d1542f84ab7b0056/ (HF facebook/dinov2-small@ed25f3a3, Apache-2.0; copy with `cp -L`);
gor_v3 = bot-5/analysis/gor_patch/gor_v3_head.npz; gor_dense = bot-5/work/gd/root/models/gor_dense/gor_dense_head.npz;
dfine_f0 = /workspace/kaggle_out/koz3-dfine-spor-full-rest/dfine_f0. A missing file shows up in the log as `GOR_DENSE_UNAVAILABLE` (output = v4, the time reads LOW).

## 4. CUDA (mandatory; without it the run doesn't count)
Before the run (from the repo root):
```
python -c "import torch;print(torch.cuda.is_available(),torch.cuda.get_device_name(0))"      # True NVIDIA GeForce GTX 1650
python bot-5/kaggle/gordense_timing/cuda_check.py .   # or the chk snippet from koz3-b5-gordense-timing.py (step 6), argument = repo root
```
The snippet uses the same code as the phase (`gor_dense.settings` → `gor_gate._device` → `load_backbone` → `.half()`, one 518² forward) and prints
`CUDA_CHECK {"device": "cuda", "param_device": "cuda:0", "dtype": "torch.float16", …}`. Need `param_device` = `cuda:0` and `dtype` = `float16`.
Checked on the box (CPU): it runs and prints `device cpu` (work/gd/sanity/cuda_check_cpu.out), so its API matches the code @86a60c5.
During the run: `nvidia-smi -l 5` → python.exe in the process list, memory used > 0, GPU load during the gor_dense phase.
In the main.py log: the line `устройство: …; cuda …` (expds/watchdog.py:300) must show cuda.

## 5. Command
```
python inference/main.py <input_folder> <out>/result.geojson  2>&1 | tee gordense_pc.log
```
Input: the same as in the PC's previous timing run (public 48 regions, bot-3/PLAN.md:147), or the train SpOr regions. NO KOZ №2 / «Сканирование» data.
Run it with the config as is (that's how the platform runs it). If the log has `мягкий потолок: … пропущено` or the watchdog fired, the guards
cut the work and the time reads low. Rerun with `EXPDS_SOFT_CEILING_S=0` and `[time] watchdog = false`, `window_deadline = false`, `budget_s = 3000` (timing only).
Optionally run OFF (`[gor_dense] enabled = false`) on the same input for the delta.

## 6. Log lines to read
- `gor_dense: a/b участков, n точек, noD удалил k объектов sp_dfine, S с`: need a == b (all SpOr regions processed); S = phase time.
- per region: `<region>: gor_dense N точек (conf …), прошло X с`.
- `фазы: … gor_dense: сырых N, X с …` (expds/watchdog.py:310), phase time.
- `готово: {json}` (inference/main.py:88): summary["phases"]["gor_dense"].seconds, plus the overall time.
- These must be ABSENT: `GOR_DENSE_UNAVAILABLE`, `SP_DFINE_UNAVAILABLE`, `Traceback`.

## 7. Pass criterion
CUDA_CHECK param_device = cuda:0 AND a == b AND no UNAVAILABLE AND **process wall time ≤ 900 s** (bot 10); ≤ 960 s is the working target.
Since the 1650 is slower than the 4090, a PC pass is enough. 900–1800 s on the PC is not a verdict: compare with the T4 kernel and the gor_dense share from `фазы`.
> 1800 s on the PC, or gor_dense not on cuda → FAIL; report to bot 5.
