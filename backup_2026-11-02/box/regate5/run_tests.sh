#!/bin/bash
export EXPDS_METRIC=/workspace/koz3/ds/e2e/score/compute_metrics.py OMP_NUM_THREADS=1
for d in $1; do cd /workspace/koz3/$d; timeout 2400 nice -n 19 /workspace/koz3/w4venv/bin/python -m pytest -q -rf -p no:cacheprovider --continue-on-collection-errors tests > /workspace/koz3/regate5/pytest_$d.log 2>&1; echo "exit $?" >> /workspace/koz3/regate5/pytest_$d.log; done
