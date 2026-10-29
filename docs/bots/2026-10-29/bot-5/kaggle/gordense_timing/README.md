# koz3-b5-gordense-timing — T4 timing kernel (bot-5). ONLY bot 1 pushes it, and only AFTER or_o2c

Pipeline only, no training. v4 (bot5-gor-dense-v4 @86a60c5) + gor_dense ON, ORGATE OFF, sp_dfine f0 (variant R, bot-7
work6/on/make_on.py) if kernel_source koz3-dfine-spor-full-rest is mounted. main.py end to end.
Input: all 132 SpOr train regions from S3 train-expds-3 (175 tif, 5.40 GB, bot-9 s3/listing.txt; no KOZ №2 / «Сканирование» data).
Proxy vs test: public test SpOr is 31 files / 30 regions / 4.68 GB (bot-8/BUDGET.md:112) → the proxy is ×1.15 by GB and
×5.6 by files, so the time is an UPPER BOUND (and the T4 is slower than the platform 4090, assumed, not measured).
For the timing run only: budget_s 3000, watchdog/window_deadline off, soft_ceiling 0 (otherwise the guards would cut work
and the time would read low). OFF = v4 as is, run if time remains → delta_on_minus_off_s.
Hard timeout 3300 s (os._exit). No GPU / DINOv2 not on cuda → FATAL, no CPU fallback.

Files: koz3-b5-gordense-timing.py (sha256 c9f9fb4456035793c0df8c84253b4ce23f3d5b50d510ad9d582f7656d1813daf),
kernel-metadata.json (sha256 3954d92a068140a8978995eb73baa43f002279bd9f0fcb5c61a114e323c27462), dataset-metadata.json.
Secrets: KOZ3_S3_ACCESS_KEY_ID / KOZ3_S3_SECRET_ACCESS_KEY (or dataset koz3-s3-cred, as in bot-9's kernels).

## Private dataset dmitrijkv/koz3-b5-gordense-timing (FLAT, 6 files, no subfolders)
| file | bytes | md5 | where to get it on the box |
|---|---|---|---|
| repo_86a60c5.tar | 865 710 080 | 5f481befaa1b0df1a43d4de321aa2a66 | `git -C /workspace/koz3_shared/bot-5/repo archive --format=tar -o DS/repo_86a60c5.tar 86a60c5` (git 2.47.3; the md5 was measured with `git archive … \| md5sum`) |
| config.json | 547 | 689053885e41f529baca946d85d9b8dd | bot-5/tmp/hf/hub/models--facebook--dinov2-small/snapshots/ed25f3a31f01632728cabb09d1542f84ab7b0056/ |
| model.safetensors | 88 249 960 | ea251be6e151ba8609ad0bc9b2e13a78 | same folder (DINOv2-S, Apache-2.0) |
| preprocessor_config.json | 436 | e40d527806779cbf0c7fd6fca552eddb | same folder |
| gor_v3_head.npz | 407 736 | acb95c52956d712a16c3dd389237a691 | bot-5/analysis/gor_patch/gor_v3_head.npz |
| gor_dense_head.npz | 204 984 | e35570e10d17edac32bea8d26fa0f5f4 | bot-5/work/gd/root/models/gor_dense/gor_dense_head.npz |

Total ≈ 954.6 MB. Copy with `cp -L` (the snapshot files are symlinks). `md5sum DS/*` must match the table, because the kernel
checks every md5 and dies on a mismatch. If the tar md5 comes out different (another git version), fix MD5["repo_86a60c5.tar"]
in the .py (the sha256 above will change). If Kaggle unpacks the tar, the kernel takes the unpacked tree (it checks for
expds/gor_dense.py). If there is no dinov2, the fallback is HF facebook/dinov2-small@ed25f3a3 (internet ON).
D-FINE f0 comes from kernel_source dmitrijkv/koz3-dfine-spor-full-rest (dfine_f0 config md5 def4fb2d…, safetensors 3fe57318…).
If it is absent, sp_dfine is OFF and the time is LOWER (see the log line "sp_dfine f0: НЕ смонтирован").

```
mkdir DS && <commands from the table> && cp dataset-metadata.json DS/
kaggle datasets create -p DS            # private by default; then, AFTER or_o2c:
kaggle kernels push -p bot-5/kaggle/gordense_timing
```
Read: /kaggle/working/run_summary.json → cuda_check.param_device == "cuda:0", steps.main_on.{wall_s, phases, phases_line,
gor_dense_summary, markers, gpu_util_max, gpu_mem_max_mib, verdict}, delta_on_minus_off_s. Logs: logs/main_on.log, main_off.log.
Targets: working ≤ 960 s, bot 10 ≤ 900 s (15 min), platform limit 1800 s.
cuda_check.py — CUDA check snippet (same as step 6 of the kernel), for the PC: `python cuda_check.py <repo_root>`.
