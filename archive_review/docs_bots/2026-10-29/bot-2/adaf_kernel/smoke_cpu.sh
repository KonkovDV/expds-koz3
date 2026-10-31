#!/bin/bash
# CPU smoke test on tiny crops (pipeline check only, not a result). 1 thread, nice, small memory.
set -eu
B2=/workspace/koz3_shared/bot-2
OUT=${1:-$B2/adaf/smoke_out}
rm -rf "$OUT"; mkdir -p "$OUT"
export MALLOC_ARENA_MAX=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 ADAF_SMOKE=1
export ADAF_SLRM=$B2/adaf/slrm ADAF_GT=/workspace/rerank/gt_train_v2.geojson ADAF_INIT=$B2/adaf/weights/OD_barrow_sd.pt
export ADAF_OUT=$OUT ADAF_TAG=smoke ADAF_ITERS=${ADAF_ITERS:-2} ADAF_BATCH=1 ADAF_CROP=256 ADAF_MIN_SIZE=256
export ADAF_TILE=256 ADAF_TSTEP=192
# two small regions from different family folds: 005_Боровичи_2 (test, fold 3) + 016_Хога (train, fold 4)
export ADAF_ONLY_REGIONS=${ADAF_ONLY_REGIONS:-005_Боровичи_2,016_Хога} ADAF_FOLD_LIST=${ADAF_FOLD_LIST:-3}
ulimit -v 2500000
# /usr/bin/time is absent on the box: peak.sh polls VmRSS of the child (nice execs python, same PID)
SECONDS=0
bash $B2/adaf/peak.sh nice -n 10 $B2/venv/bin/python $B2/adaf_kernel/adaf_finetune.py > "$OUT/log.txt" 2>&1 || { tail -30 "$OUT/log.txt"; exit 1; }
tail -1 "$OUT/log.txt"; echo "elapsed_s $SECONDS"
$B2/venv/bin/python $B2/adaf_kernel/check_maps.py "$OUT"
