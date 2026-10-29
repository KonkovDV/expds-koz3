#!/bin/bash
# bot-6 ORSP v2: повтор методики bot-10 work9/run9c.sh (global, k10/lofo/seeds); Af/orspF/orsp — кривые bot-10 (воспроизведение −0,00169).
cd /workspace/koz3_shared/bot-6/work/orsp2
B=/workspace/koz3_shared/bot-10
export EXPDS_METRIC=$B/repo/materials/compute_metrics.py OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 GDAL_NUM_THREADS=2 GDAL_CACHEMAX=128 TMPDIR=/workspace/koz3_shared/bot-6/work/pytest_tmp
PY=/workspace/rerank/venv/bin/python; G=/workspace/rerank/gt_train_v2.geojson
L="prlimit --as=2000000000 nice -n 10"
echo "start $(date +%T)"
$L $PY orspR_fam.py fam_orspR > fam_orspR.log 2>&1; echo "fam rc=$? $(date +%T)"
EXPDS_MATCH_SCOPE=global $L $PY $B/repo/scripts/family_curves.py --fam fam_orspR --gt $G --clusters $B/work/family_clusters.json --out curves_orspR_g > curves_orspR_g.log 2>&1; echo "c rc=$? $(date +%T)"
$L $PY $B/work4/paired.py --pols drop,keep --base A=$B/work/curves --b Af=$B/work9/curves_Af_g --b orspF=$B/work9/curves_orspF_g --b orsp=$B/work9/curves_orsp_g --b orspR=curves_orspR_g --out orspR_global.json > orspR_global.log 2>&1; echo "p rc=$? $(date +%T)"
echo done
