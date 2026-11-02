#!/bin/bash
# lane runner: ./run_replays.sh v1 v2 ...
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
for v in "$@"; do
  /workspace/koz3/w4venv/bin/python s2_replay.py $v /workspace/koz3/w1/satrec/out/out_$v.geojson > s2_$v.log 2>&1 || echo "FAIL $v" >> s2_fail.log
done
