#!/bin/bash
cd /workspace/koz3_shared/bot-9/kaggle/convnext_lidar/c3sc_f23_out; export OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1; ulimit -v 1800000
for f in 2 3; do nice -n 10 /workspace/rerank/venv/bin/python score_thr.py $f > metric/score_thr_f$f.log 2>&1; echo "f$f rc=$? $(date +%T)"; done
