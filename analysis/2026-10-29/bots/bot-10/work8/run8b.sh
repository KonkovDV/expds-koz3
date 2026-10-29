#!/bin/bash
cd /workspace/koz3_shared/bot-10
export EXPDS_METRIC=$PWD/repo/materials/compute_metrics.py OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
PY=/workspace/rerank/venv/bin/python; G=/workspace/rerank/gt_train_v2.geojson
ulimit -v 3000000
echo "start $(date +%T)"
export EXPDS_MATCH_SCOPE=site
D=work8/curves_s_S; rm -rf $D; mkdir -p $D
for f in work4/curves_site/*.pkl; do ln -s $(readlink -f $f) $D/$(basename $f); done
rm -f $D/Гнёздово.pkl $D/Семибратнее.pkl $D/Судьбищи.pkl
nice -n 10 $PY repo/scripts/family_curves.py --fam work7/fam_S --gt $G --clusters work/family_clusters.json --out $D > $D.log 2>&1
echo "site curves $(ls $D|wc -l) $(date +%T)"
nice -n 10 $PY work4/paired.py --boot 200 --base A=work4/curves_site --b S=$D --out work8/paired_S_site.json > work8/paired_S_site.log 2>&1
echo "S site done $(date +%T)"
