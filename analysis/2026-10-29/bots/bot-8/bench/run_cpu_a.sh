#!/bin/bash
# Прогон bench_cpu_a.py по участкам, 2 потока, nice 10, taskset.
source /workspace/koz3_shared/speedup/env.sh
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 RAYON_NUM_THREADS=2 EXPDS_THREADS=2 PYTHONDONTWRITEBYTECODE=1
D=/workspace/data/train-expds-3
CPUS=${CPUS:-6,7}
for s in "$@"; do
  f=$(ls $D/$s/*/*.laz | head -1)
  extra=""; [ "$s" = "021_Новенькое" ] && extra="--force-pmf"
  nice -n 10 taskset -c $CPUS python bench_cpu_a.py "$f" cpu_a/$s.json --reps ${REPS:-3} $extra || echo FAIL $s
done
rm -rf /tmp/koz3b8_cpu_a
