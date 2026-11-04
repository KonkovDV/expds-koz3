#!/bin/bash
# W1-SATDET track 1 gate after the Kaggle kernel finished (PREREG 843d807). nice 10, <=4 threads.
# 1) pull outputs (no weights are produced)  2) nested prep D1/D2  3) prod replays  4) score base P1 P2 D1 D2 (m=4, Bonf 0.975)
set -euo pipefail
K=/workspace/koz3; W=$K/w1/satdet; PY=$K/w4venv/bin/python; cd "$(dirname "$0")"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
mkdir -p $W/pulled && $K/orch/pool_kaggle.sh elenakonkova kernels output elenakonkova/koz3-w1-satdet-frcnn -p $W/pulled
nice -n 10 $PY t1_prep.py --det $W/pulled/oof_frcnn_spor.geojson > $W/out/t1_det.log 2>&1
for v in D1 D2; do nice -n 10 $PY t2_replay.py $v $W/out/out_$v.geojson $W/out/prep_det.pkl > $W/out/s2_$v.log 2>&1; done
nice -n 10 $PY t3_score.py base P1 P2 D1 D2 > $W/out/s3_all.log 2>&1
grep -E "delta|PASS" $W/out/s3_all.log
