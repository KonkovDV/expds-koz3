#!/bin/bash
# smoke (голова centerhead), читающий готовый prep-выход (_scratch/dry/prep) — путь kernel_sources после смены головы.
set -a; . ~/.koz3s3/env; set +a
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 RAYON_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
D=/workspace/koz3_shared/bot-9/kaggle/convnext_lidar
export CNX_CACHE=$D/_scratch/s3cache TORCH_HOME=$D/_scratch/torch_home HF_HOME=$D/_scratch/hf_home CNX_REGIONS="191_Осечки_1.3км,016_Хога"
ulimit -v 8000000
CNX_PREP_DIR=$D/_scratch/dry/prep/cnx_relief CNX_BUILD_RELIEF=0 CNX_OUT=$D/_scratch/dry/out_prep CNX_RELIEF_DIR=$D/_scratch/relief_q \
CNX_CROP=128 CNX_ITERS=2 CNX_BS=2 CNX_TTA=1 CNX_WARMUP=1 CNX_JOBS="c3-sc:3" nice -n 10 $D/_scratch/venv/bin/python $D/koz3-cnx-li-smoke/koz3-cnx-li-smoke.py --allow-cpu
