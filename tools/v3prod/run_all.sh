#!/bin/bash
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
cd /workspace/koz3/orch/bot-W1/work/v3prod
for m in PH PC PR; do
  nice -n 10 /workspace/koz3/w4venv/bin/python a_replay.py v2c $m /workspace/koz3/w1/v3audit/out_v2c_$m.geojson > a_v2c_$m.log 2>&1 &
  nice -n 10 /workspace/koz3/w4venv/bin/python a_replay.py v3 $m /workspace/koz3/w1/v3audit/out_v3_$m.geojson > a_v3_$m.log 2>&1
  wait
done
nice -n 10 /workspace/koz3/venv/bin/python b_score.py S0 PH PC PR > b_score.log 2>&1
echo ALLDONE >> b_score.log
