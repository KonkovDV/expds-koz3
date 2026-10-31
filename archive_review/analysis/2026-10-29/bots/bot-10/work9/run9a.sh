#!/bin/bash
cd /workspace/koz3_shared/bot-10/work9
export EXPDS_METRIC=/workspace/koz3_shared/bot-10/repo/materials/compute_metrics.py OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
ulimit -v 3000000
PY=/workspace/rerank/venv/bin/python
echo "start $(date +%T)"
EXPDS_MATCH_SCOPE=global nice -n 10 $PY geo_exact.py ex_global.pkl > ex_global.log 2>&1; echo "global rc=$? $(date +%T)"
nice -n 10 $PY geo_stats.py ex_global.pkl ex_global_stats.json > ex_global_stats.log 2>&1
EXPDS_MATCH_SCOPE=site CURV_A=/workspace/koz3_shared/bot-10/work4/curves_site CURV_S=/workspace/koz3_shared/bot-10/work8/curves_s_S nice -n 10 $PY geo_exact.py ex_site.pkl > ex_site.log 2>&1; echo "site rc=$? $(date +%T)"
nice -n 10 $PY geo_stats.py ex_site.pkl ex_site_stats.json > ex_site_stats.log 2>&1
echo "done $(date +%T)"
