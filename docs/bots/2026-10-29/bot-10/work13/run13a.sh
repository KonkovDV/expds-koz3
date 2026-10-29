#!/bin/bash
cd /workspace/koz3_shared/bot-10/work13
export EXPDS_METRIC=/workspace/koz3_shared/bot-10/repo/materials/compute_metrics.py OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 EXPDS_MATCH_SCOPE=global
PY=/workspace/rerank/venv/bin/python
ulimit -v 3000000
echo "start $(date +%T)"
[ -f taus_S.json ] || nice -n 10 $PY ../work6/taus.py S /workspace/koz3_shared/bot-10/work7/curves_g_S > taus_S.log 2>&1; echo "taus rc=$? $(date +%T)"
nice -n 10 $PY lbv2_oof.py keep lbv2_keep.pkl v2,A > lbv2_keep.log 2>&1; echo "keep rc=$? $(date +%T)"
nice -n 10 $PY lbv2_oof.py drop lbv2_drop.pkl v2 > lbv2_drop.log 2>&1; echo "drop rc=$? $(date +%T)"
echo done
