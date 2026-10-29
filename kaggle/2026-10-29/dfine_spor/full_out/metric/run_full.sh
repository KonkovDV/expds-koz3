#!/bin/bash
# Полный набор семей: база A = bot-10/work/curves; варианты = те же кривые, где 71 семья фолдов 4,2,1 заменена на «база + D-FINE».
M=/workspace/koz3_shared/bot-9/kaggle/dfine_spor/full_out/metric; B10=/workspace/koz3_shared/bot-10
export EXPDS_METRIC=$B10/repo/materials/compute_metrics.py OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 EXPDS_MATCH_SCOPE=global
ulimit -v 3000000; cd $B10
nice -n 10 /workspace/rerank/venv/bin/python work4/paired.py --pols drop,keep --base A=work/curves --b C=$M/curves_full_C --b E=$M/curves_full_E --out $M/paired_full.json > $M/paired_full.log 2>&1; echo "p rc=$? $(date +%T)"
