#!/bin/bash
# грубый проход городищ по одному участку; пропускает готовые. ≤2 потока, nice 10.
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 NUMBA_NUM_THREADS=2 RAYON_NUM_THREADS=1 POLARS_MAX_THREADS=2
B=/workspace/koz3_shared/bot-4
cd $B/work/dem
for p in $(find /workspace/data/train-expds-3 -name "*.la[sz]" -printf "%s %p\n" | sort -n | cut -d' ' -f2-); do
  reg=$(echo "$p" | cut -d/ -f5)
  [ -f out/$reg.geojson ] && continue
  mkdir -p roots/$reg && ln -sfn /workspace/data/train-expds-3/$reg roots/$reg/$reg
  s=$(date +%s)
  ( ulimit -v 6000000; nice -n 10 taskset -c 0,1 $B/venv/bin/python $B/repo/scripts/hillfort_oof.py --root roots/$reg --out out/$reg.tmp --preset coarse ) > logs_$reg.txt 2>&1
  rc=$?
  [ $rc -eq 0 ] && mv out/$reg.tmp out/$reg.geojson
  echo "$(date +%T) $reg rc=$rc $(( $(date +%s)-s ))s $(tail -1 logs_$reg.txt | cut -c1-120)"
done
echo ALLDONE
