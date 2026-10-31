#!/bin/bash
cd /workspace/koz3_shared/bot-10
export EXPDS_METRIC=$PWD/repo/materials/compute_metrics.py OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
PY=/workspace/rerank/venv/bin/python; G=/workspace/rerank/gt_train_v2.geojson
ulimit -v 3000000
echo "start $(date +%T)"
nice -n 10 $PY work9/orsp_fam.py --src work/fam --dst work9/fam_orsp > work9/orsp_fam.log 2>&1; echo "fam rc=$? $(date +%T)"
EXPDS_MATCH_SCOPE=global nice -n 10 $PY repo/scripts/family_curves.py --fam work9/fam_orsp --gt $G --clusters work/family_clusters.json --out work9/curves_orsp_g > work9/curves_orsp_g.log 2>&1; echo "cg rc=$? $(date +%T)"
nice -n 10 $PY work4/paired.py --pols drop,keep --base A=work/curves --b orsp=work9/curves_orsp_g --out work9/orsp_global.json > work9/orsp_global.log 2>&1; echo "pg rc=$? $(date +%T)"
EXPDS_MATCH_SCOPE=site nice -n 10 $PY repo/scripts/family_curves.py --fam work9/fam_orsp --gt $G --clusters work/family_clusters.json --out work9/curves_orsp_s > work9/curves_orsp_s.log 2>&1; echo "cs rc=$? $(date +%T)"
nice -n 10 $PY work4/paired.py --pols drop,keep --base A=work4/curves_site --b orsp=work9/curves_orsp_s --out work9/orsp_site.json > work9/orsp_site.log 2>&1; echo "ps rc=$? $(date +%T)"
echo done
cd /workspace/koz3_shared/bot-10/work9
nice -n 10 $PY taus.py orsp /workspace/koz3_shared/bot-10/work9/curves_orsp_g > taus_orsp.log 2>&1; echo "taus rc=$? $(date +%T)"
export OMP_NUM_THREADS=1
nice -n 10 $PY twins9.py global twins9_global.pkl > twins9_global.log 2>&1; echo "tw rc=$? $(date +%T)"
nice -n 10 $PY stats9.py twins9_global.pkl twins9_stats.json > twins9_stats.log 2>&1; echo "st rc=$? $(date +%T)"
echo done2
