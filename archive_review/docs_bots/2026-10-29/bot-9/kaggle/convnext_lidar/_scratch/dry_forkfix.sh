#!/bin/bash
# CPU dry-run на боксе: smoke-ядро как есть, крошечные кропы/итерации. Ключи — из ~/.koz3s3/env (не печатаются).
set -a; . ~/.koz3s3/env; set +a
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 RAYON_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
D=/workspace/koz3_shared/bot-9/kaggle/convnext_lidar
export CNX_OUT=${CNX_OUT:-$D/_scratch/dry_ff/out} CNX_CACHE=$D/_scratch/s3cache CNX_RELIEF_DIR=$D/_scratch/relief CNX_KEEP_LAS=1
export CNX_CROP=${CNX_CROP:-128} CNX_ITERS=${CNX_ITERS:-3} CNX_BS=${CNX_BS:-2} CNX_TTA=${CNX_TTA:-2} CNX_WARMUP=2 CNX_PASTE_BANK=200 CNX_LOG_EVERY=1
export TORCH_HOME=$D/_scratch/torch_home HF_HOME=$D/_scratch/hf_home
ulimit -v ${ULIM:-2000000}
exec nice -n 10 $D/_scratch/venv/bin/python $D/_scratch/dry_forkfix_smoke.py --allow-cpu
