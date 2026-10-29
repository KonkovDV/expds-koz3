#!/bin/bash
# CPU dry-run prep-ядра (2 участка) и smoke, читающего prep-выход (путь kernel_sources).
set -a; . ~/.koz3s3/env; set +a
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 RAYON_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
D=/workspace/koz3_shared/bot-9/kaggle/convnext_lidar
export CNX_CACHE=$D/_scratch/s3cache CNX_KEEP_LAS=1 TORCH_HOME=$D/_scratch/torch_home HF_HOME=$D/_scratch/hf_home
export CNX_REGIONS="191_Осечки_1.3км,016_Хога"
ulimit -v 8000000
CNX_OUT=$D/_scratch/dry/prep CNX_RELIEF_DIR=$D/_scratch/relief_p nice -n 10 $D/_scratch/venv/bin/python $D/koz3-cnx-li-prep/koz3-cnx-li-prep.py || exit 1
cmp $D/_scratch/dry/prep/cnx_relief/016_Хога.npy $D/_scratch/relief/016_Хога.npy && echo "prep == smoke-сборка (016_Хога.npy побайтно)"
CNX_PREP_DIR=$D/_scratch/dry/prep/cnx_relief CNX_BUILD_RELIEF=0 CNX_OUT=$D/_scratch/dry/out_prep CNX_RELIEF_DIR=$D/_scratch/relief_q \
CNX_CROP=128 CNX_ITERS=2 CNX_BS=2 CNX_TTA=1 CNX_WARMUP=1 CNX_JOBS="c3-sc:3" nice -n 10 $D/_scratch/venv/bin/python $D/koz3-cnx-li-smoke/koz3-cnx-li-smoke.py --allow-cpu
