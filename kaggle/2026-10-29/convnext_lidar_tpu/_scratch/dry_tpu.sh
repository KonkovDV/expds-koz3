#!/bin/bash
# CPU dry-run TPU-ядра на боксе. $1 = cpu (torch CPU, 2 процесса-«чипа», путь оркестрации) | xla (PJRT CPU, 1 процесс, маленький энкодер)
set -a; . ~/.koz3s3/env; set +a          # ключи S3 только в env процесса, не печатаются
D=/workspace/koz3_shared/bot-9/kaggle/convnext_lidar_tpu; S=$D/_scratch
export CNX_CACHE=$S/s3cache CNX_PREP_DIR=$S/prep/cnx_relief CNX_BUILD_RELIEF=0 CNX_REGIONS="191_Осечки_1.3км,016_Хога" HF_HOME=$S/hf_home
export CNX_CROP=128 CNX_STRIDE=128 CNX_BS=2 CNX_TTA=1 CNX_WARMUP=1 CNX_WORKERS=0 CNX_LOG_EVERY=2 CNX_ENCODER_TEST=tu-convnext_atto CNX_INFER_BS=4
if [ "$1" = cpu ]; then
  export CNX_TPU_DEV=cpu CNX_TPU_PROCS=2 CNX_ITERS=6 CNX_JOBS="${CNX_JOBS_DRY:-c5-sc:4}" CNX_OUT=$S/dry/out_cpu CNX_RELIEF_DIR=$S/relief_cpu
else
  export CNX_TPU_DEV=xla PJRT_DEVICE=CPU CPU_NUM_DEVICES=1 CNX_TPU_PROCS=1 CNX_ITERS=10 CNX_JOBS="c3-sc:3" CNX_OUT=$S/dry/out_xla CNX_RELIEF_DIR=$S/relief_xla
fi
exec $S/venv/bin/python $D/koz3-cnx-li-tpu-smoke/koz3-cnx-li-tpu-smoke.py --allow-cpu
