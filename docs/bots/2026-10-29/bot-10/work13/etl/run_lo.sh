#!/bin/bash
cd /workspace/koz3_shared/bot-10
export EXPDS_METRIC=$PWD/repo/materials/compute_metrics.py OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 EXPDS_MATCH_SCOPE=global
PY=/workspace/rerank/venv/bin/python; G=/workspace/rerank/gt_train_v2.geojson
ulimit -v 3000000
D=work13/etl/c_Alo_1; rm -rf $D; mkdir -p $D
for f in work/curves/*.pkl; do ln -s $(readlink -f $f) $D/$(basename $f); done; rm -f $D/Гнёздово.pkl
EXPDS_ETGR_LAYER=1 nice -n 10 $PY repo/scripts/family_curves.py --fam work13/etl/fam_Alo --gt $G --clusters work/family_clusters.json --out $D > $D.log 2>&1; echo "curves rc=$? $(date +%T)"
rm -rf work13/etl/fam_Alo
nice -n 10 $PY work4/paired.py --base Alo=$D --b S1=work13/etl/c_S_1 --b Ss6_1=work13/etl/c_Ss6_1 --boot 500 --out work13/etl/paired_layer_vsAlo.json > work13/etl/paired_layer_vsAlo.log 2>&1; echo "p rc=$? $(date +%T)"
