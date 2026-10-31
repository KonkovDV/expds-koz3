#!/bin/bash
# lofo --scope site (G), после run2/run3
cd /workspace/koz3_shared/bot-6/work/orgate; export OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 TMPDIR=/workspace/koz3_shared/bot-6/work/pytest_tmp
while pgrep -f "run2.sh|run3.sh" >/dev/null; do sleep 20; done
prlimit --as=2000000000 nice -n 10 /workspace/rerank/venv/bin/python gate_eval.py --layout lofo --scope site --variants G > res/lofo_site.log 2>&1; echo "done $(date +%T)"
