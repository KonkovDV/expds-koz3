#!/bin/bash
# Доп. 5: FR1h exact (порядок сдачи), global
cd /workspace/koz3_shared/bot-10/work11
export EXPDS_METRIC=/workspace/koz3_shared/bot-10/repo/materials/compute_metrics.py OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
PY=/workspace/rerank/venv/bin/python
ulimit -v 3000000
echo "start $(date +%T)"
while pgrep -f "run11j.sh" >/dev/null; do sleep 15; done; echo "run11j gone $(date +%T)"
test -s taus_FR1h.json || { echo "no taus_FR1h.json"; exit 1; }
nice -n 10 $PY fr_exact.py global ../work6/taus_A.json taus_FR1h.json fr_exact_global.pkl > fr_exact_global.log 2>&1; echo "ex rc=$? $(date +%T)"
nice -n 10 $PY stats11.py fr_exact_global.pkl fr_exact_global.json FR-A,TOFR-TO,TO-A,TOFR-A > fr_exact_stats.log 2>&1; echo "st rc=$? $(date +%T)"
echo done
