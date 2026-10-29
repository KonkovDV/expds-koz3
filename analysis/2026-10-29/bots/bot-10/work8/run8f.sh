#!/bin/bash
cd /workspace/koz3_shared/bot-10/work8
export EXPDS_METRIC=/workspace/koz3_shared/bot-10/repo/materials/compute_metrics.py OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
ulimit -v 3000000
PY=/workspace/rerank/venv/bin/python
echo "start $(date +%T)"
nice -n 10 $PY twins78.py global twins78_global.pkl > twins78_global.log 2>&1; echo "rc=$? $(date +%T)"
nice -n 10 $PY ../work6/stats_twins.py twins78_global.pkl twins78_stats.json > twins78_stats.log 2>&1; echo "stats rc=$? $(date +%T)"
