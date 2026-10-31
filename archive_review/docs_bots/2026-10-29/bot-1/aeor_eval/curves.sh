#!/bin/bash
# $1 = variant name (fam_$1 -> curves_$1)
cd /workspace/koz3_shared/bot-1/aeor_eval
B=/workspace/koz3_shared/bot-10
export EXPDS_METRIC=$B/repo/materials/compute_metrics.py OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 EXPDS_MATCH_SCOPE=global
ulimit -v 4000000
nice -n 10 /workspace/rerank/venv/bin/python $B/repo/scripts/family_curves.py --fam fam_$1 --gt gt_t.geojson --clusters $B/work/family_clusters.json --out curves_$1
