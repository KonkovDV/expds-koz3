#!/bin/bash
# CPU smoke test on tiny crops (pipeline check only, not a result). 1 thread, nice, small memory.
set -eu
B2=/workspace/koz3_shared/bot-2
OUT=${1:-/tmp/b2_adaf_smoke}
rm -rf "$OUT"; mkdir -p "$OUT"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 ADAF_SMOKE=1
export ADAF_SLRM=$B2/adaf/slrm ADAF_GT=/workspace/rerank/gt_train_v2.geojson ADAF_INIT=$B2/adaf/weights/OD_barrow_sd.pt
export ADAF_OUT=$OUT ADAF_TAG=smoke ADAF_ITERS=${ADAF_ITERS:-2} ADAF_BATCH=1 ADAF_CROP=256 ADAF_MIN_SIZE=256
export ADAF_TILE=256 ADAF_TSTEP=192
# two small regions from different family folds: 005_Боровичи_2 (test, fold 3) + 016_Хога (train, fold 4)
export ADAF_ONLY_REGIONS=${ADAF_ONLY_REGIONS:-005_Боровичи_2,016_Хога} ADAF_FOLD_LIST=${ADAF_FOLD_LIST:-3}
ulimit -v 3000000
/usr/bin/time -v nice -n 10 $B2/venv/bin/python $B2/adaf_kernel/adaf_finetune.py 2> "$OUT/time.txt"
grep -E "Maximum resident|Elapsed" "$OUT/time.txt"
$B2/venv/bin/python $B2/adaf_kernel/check_maps.py "$OUT"
