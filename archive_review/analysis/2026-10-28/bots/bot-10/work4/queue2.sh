#!/bin/bash
cd /workspace/koz3_shared/bot-10
export EXPDS_METRIC=$PWD/repo/materials/compute_metrics.py OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
PY=/workspace/rerank/venv/bin/python; G=/workspace/rerank/gt_train_v2.geojson; B7=/workspace/koz3_shared/bot-7/work2
ulimit -v 3000000
while kill -0 126513 2>/dev/null; do sleep 10; done
echo "start $(date +%T)"
nice -n 10 $PY work4/paired.py --pols drop --base A=work/curves --b kur=$B7/curves_kur --b arch=$B7/curves_arch --b bundle=$B7/curves_bundle --out work4/bot7_global.json > work4/bot7_global.log 2>&1
echo "bot7 global $(date +%T)"
for v in kur arch bundle; do EXPDS_MATCH_SCOPE=site nice -n 10 $PY repo/scripts/family_curves.py --fam $B7/fam_$v --gt $G --clusters work/family_clusters.json --out work4/curves_site_b7$v > work4/curves_site_b7$v.log 2>&1; done
echo "site curves $(date +%T)"
nice -n 10 $PY work4/paired.py --pols drop --base A=work4/curves_site --b kur=work4/curves_site_b7kur --b arch=work4/curves_site_b7arch --b bundle=work4/curves_site_b7bundle --out work4/bot7_site.json > work4/bot7_site.log 2>&1
echo "done $(date +%T)"
