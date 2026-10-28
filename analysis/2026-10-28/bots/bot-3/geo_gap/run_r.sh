#!/bin/bash
# Растровые варианты object_poly (ветка bot3-geophys) на 201–217, все модальности. 1 поток, nice 10, ulimit -v 3 ГБ.
set -u
cd /workspace/koz3_shared/bot-3/repo
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
ulimit -v 3000000
OUT=/workspace/koz3_shared/bot-3/geo_gap/r
REGS="${REGS:-$(ls /workspace/data/train-expds-3 | grep -E '^2(0[1-9]|1[0-7])_' | tr '\n' ' ')}"
COMMON="--root /workspace/data/train-expds-3"
run() { name=$1; shift
  nice -n 10 ../venv/bin/python scripts/geo_predict.py $COMMON --out $OUT/$name.geojson "$@" --regions $REGS 2>&1 | grep -v 'рамка LAS' > $OUT/$name.log
  echo "$name exit=${PIPESTATUS[0]}" >> $OUT/done.txt; }
for c in ${CFGS:-RB R1 R2 R3}; do case $c in
 RB) run RB ;;
 R1) run R1 --set 'geophys.split="dt"' --set geophys.split_h=0.5 --set geophys.pole_circle_m=0.05 ;;
 R2) run R2 --set 'geophys.split="dt"' --set geophys.split_h=0.5 ;;
 R3) run R3 --set geophys.pole_circle_m=0.05 ;;
 R4) run R4 --set geophys.etgr.layer_copy=true ;;
esac; done
