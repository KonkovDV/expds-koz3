#!/bin/bash
# bot-6 ORSP rework (orsp3): новый код (bot-6/repo HEAD) → fam_orsp3 (без полов) → кривые → paired vs A (+ orsp v1 bot-10 для сверки) → per-fold пороги
cd /workspace/koz3_shared/bot-6/work/orsp3
B=/workspace/koz3_shared/bot-10; ME=/workspace/koz3_shared/bot-6/work/orsp3
export EXPDS_METRIC=$B/repo/materials/compute_metrics.py OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 GDAL_NUM_THREADS=2 GDAL_CACHEMAX=128 TMPDIR=/workspace/koz3_shared/bot-6/work/pytest_tmp
PY=/workspace/rerank/venv/bin/python; G=/workspace/rerank/gt_train_v2.geojson
L="prlimit --as=2000000000 nice -n 10"
echo "start $(date +%T) repo $(git -C ../../repo rev-parse --short HEAD)"
$L $PY orsp3_fam.py fam_orsp3 > fam_orsp3.log 2>&1; echo "fam rc=$? $(date +%T)"
EXPDS_MATCH_SCOPE=global $L $PY $B/repo/scripts/family_curves.py --fam fam_orsp3 --gt $G --clusters $B/work/family_clusters.json --out curves_orsp3_g > curves_orsp3_g.log 2>&1; echo "c rc=$? $(date +%T)"
EXPDS_MATCH_SCOPE=global $L $PY $B/repo/scripts/family_curves.py --fam $B/work11/fam0 --gt $G --clusters $B/work/family_clusters.json --out curves_A0_g > curves_A0_g.log 2>&1; echo "cA0 rc=$? $(date +%T)"
cd $B && $L $PY work4/paired.py --pols drop,keep --base A=$B/work/curves --b A0=$ME/curves_A0_g --b orsp_v1=$B/work9/curves_orsp_g --b orsp3=$ME/curves_orsp3_g --out $ME/orsp3_global.json > $ME/orsp3_global.log 2>&1; echo "p rc=$? $(date +%T)"
cd $ME && $L $PY taus_fold.py $ME/curves_orsp3_g > taus_fold.log 2>&1; echo "t rc=$? $(date +%T)"
echo done
