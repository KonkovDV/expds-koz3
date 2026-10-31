#!/bin/bash
cd /workspace/koz3_shared/bot-10/work13
export EXPDS_METRIC=/workspace/koz3_shared/bot-10/repo/materials/compute_metrics.py OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 EXPDS_MATCH_SCOPE=global
ulimit -v 3000000
while pgrep -f "run13b" >/dev/null; do sleep 10; done
nice -n 10 /workspace/rerank/venv/bin/python ae_diag.py keep ae_diag_keep.json v2 > ae_diag.log 2>&1; echo "rc=$? $(date +%T)"
