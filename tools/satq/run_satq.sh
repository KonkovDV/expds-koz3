#!/bin/bash
# W1-SATQ gate (PREREG docs/PREREG_W1_SATQ.md): 1) gate-side nested prep  2) prod replays  3) score (m=5, Bonf 0.98, Q1 0.99). nice 10, <=4 threads.
set -euo pipefail
K=/workspace/koz3; W=$K/w1/satq/out; PY=$K/w4venv/bin/python; cd "$(dirname "$0")"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
[ -f $W/prep.pkl ] || nice -n 10 $PY q1_prep.py > $W/q1_prep.log 2>&1
for v in base Q1a Q1b Q1c FP1 FP2; do nice -n 10 $PY q2_replay.py $v $W/out_$v.geojson > $W/s2_$v.log 2>&1; done
nice -n 10 $PY q3_score.py base Q1a Q1b Q1c FP1 FP2 > $W/s3_all.log 2>&1
grep -E "delta|PASS" $W/s3_all.log
