# v3 (164707a, sp_dfine strict) vs v2 eb268d6 — NOT RUN (29.10 10:30 MSK, bot-1)
PC local-exec blocked (spawn EPERM; C:\Sloi outside allowed root) right after the 10:21 reboot. Nothing built or run on PC.
Box-side prep: main 164707a cloned (/tmp/k3). configs/default.toml: [sp_dfine] enabled=false; enabling needs
weights = ["models/dfine_final"] (config.json + model.safetensors, fp32; f0 test = models/dfine_f0), drop_groups (variant R),
[thresholds_by_markup.sp_dfine] (bot-9 thr/sweep.log) and [thresholds_by_markup.or] arkhitektury = 0.483891; strict preflight
checks transformers==5.21.0 + DFine classes + weights + thresholds. Whether models/dfine_f0|dfine_final exist on PC: unverified.
