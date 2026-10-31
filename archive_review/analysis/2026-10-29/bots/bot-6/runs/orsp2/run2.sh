#!/bin/bash
# Контроль утечки пола: AF2 = A с полами = all-data пороги A (sp_focal 0,364475/0,372525 = сдача sp_or; sp_focal2 0,501645/0,645505).
cd /workspace/koz3_shared/bot-6/work/orsp2
B=/workspace/koz3_shared/bot-10; O=$PWD
export EXPDS_METRIC=$B/repo/materials/compute_metrics.py OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 GDAL_NUM_THREADS=2 GDAL_CACHEMAX=128 TMPDIR=/workspace/koz3_shared/bot-6/work/pytest_tmp
PY=/workspace/rerank/venv/bin/python; G=/workspace/rerank/gt_train_v2.geojson; L="prlimit --as=2000000000 nice -n 10"
echo "start $(date +%T)"
$L $PY aF2_fam.py fam_AF2 > fam_AF2.log 2>&1; echo "fam rc=$? $(date +%T)"
EXPDS_MATCH_SCOPE=global $L $PY $B/repo/scripts/family_curves.py --fam fam_AF2 --gt $G --clusters $B/work/family_clusters.json --out curves_AF2_g > curves_AF2_g.log 2>&1; echo "c rc=$? $(date +%T)"
cd $B
$L $PY work4/paired.py --pols drop --base AF2=$O/curves_AF2_g --b orsp=work9/curves_orsp_g --b orspR=$O/curves_orspR_g --b A=work/curves --out $O/vsAF2_global.json > $O/vsAF2_global.log 2>&1; echo "p rc=$? $(date +%T)"
echo done
