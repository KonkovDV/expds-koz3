#!/bin/bash
cd /workspace/koz3_shared/bot-10
export EXPDS_METRIC=$PWD/repo/materials/compute_metrics.py OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
PY=/workspace/rerank/venv/bin/python; G=/workspace/rerank/gt_train_v2.geojson
ulimit -v 3000000
echo "start $(date +%T)"
nice -n 10 $PY work9/orsp_fam.py --src work/fam --dst work9/fam_Af --floor --nofuse > work9/fam_Af.log 2>&1; echo "Af rc=$? $(date +%T)"
nice -n 10 $PY work9/orsp_fam.py --src work/fam --dst work9/fam_orspF --floor > work9/fam_orspF.log 2>&1; echo "orspF rc=$? $(date +%T)"
for v in Af orspF; do EXPDS_MATCH_SCOPE=global nice -n 10 $PY repo/scripts/family_curves.py --fam work9/fam_$v --gt $G --clusters work/family_clusters.json --out work9/curves_${v}_g > work9/curves_${v}_g.log 2>&1; echo "c $v rc=$? $(date +%T)"; done
nice -n 10 $PY work4/paired.py --pols drop,keep --base A=work/curves --b Af=work9/curves_Af_g --b orspF=work9/curves_orspF_g --b orsp=work9/curves_orsp_g --out work9/orspF_global.json > work9/orspF_global.log 2>&1; echo "p rc=$? $(date +%T)"
echo done
