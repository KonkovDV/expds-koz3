#!/bin/bash
M=/workspace/koz3_shared/bot-9/kaggle/convnext_lidar/c3sc_f23_out/metric; B10=/workspace/koz3_shared/bot-10
export EXPDS_METRIC=$B10/repo/materials/compute_metrics.py OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 EXPDS_MATCH_SCOPE=global
PY=/workspace/rerank/venv/bin/python; ulimit -v 3000000
cd $B10 && nice -n 10 $PY work4/paired.py --pols drop,keep --base A=$M/c_A --b X=$M/c_X --b Xx=$M/c_Xx --b X4=$M/c4_X --b Xx4=$M/c4_Xx --out $M/paired.json > $M/paired.log 2>&1; echo "paired rc=$? $(date +%T)"
cd $M && nice -n 10 $PY fam_delta.py > fam_delta.log 2>&1; echo "fam_delta rc=$? $(date +%T)"
