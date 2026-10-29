#!/bin/bash
cd /workspace/koz3_shared/bot-10/work11
while pgrep -f run11a.sh >/dev/null; do sleep 20; done
export EXPDS_METRIC=/workspace/koz3_shared/bot-10/repo/materials/compute_metrics.py OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
PY=/workspace/rerank/venv/bin/python; ulimit -v 3000000
echo "start $(date +%T)"
nice -n 10 $PY gm_exact.py global ../work6/taus_A.json gm_exact_global.pkl > gm_exact_global.log 2>&1; echo "eg rc=$? $(date +%T)"
nice -n 10 $PY stats11.py gm_exact_global.pkl gm_exact_global_stats.json > gm_exact_global_stats.log 2>&1; echo "sg rc=$? $(date +%T)"
nice -n 10 $PY gm_exact.py site taus_A_site.json gm_exact_site.pkl > gm_exact_site.log 2>&1; echo "es rc=$? $(date +%T)"
nice -n 10 $PY stats11.py gm_exact_site.pkl gm_exact_site_stats.json > gm_exact_site_stats.log 2>&1; echo "ss rc=$? $(date +%T)"
echo done
