#!/bin/bash
# bot-10: ORSP v2 (bot-6 c050028, код bot-6/repo) БЕЗ полов (вложенный sweep решает сам) — честный эквивалент orspR.
cd /workspace/koz3_shared/bot-10
export EXPDS_METRIC=$PWD/repo/materials/compute_metrics.py OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 NOFLOOR=1
PY=/workspace/rerank/venv/bin/python; G=/workspace/rerank/gt_train_v2.geojson; W=work10/orsp
L="prlimit --as=2200000000 nice -n 10"
echo "start $(date +%T)"
$L $PY $W/orspN_fam.py $W/fam_orspN > $W/fam_orspN.log 2>&1; echo "fam rc=$? $(date +%T) $(tail -1 $W/fam_orspN.log)"
EXPDS_MATCH_SCOPE=global $L $PY repo/scripts/family_curves.py --fam $W/fam_orspN --gt $G --clusters work/family_clusters.json --out $W/curves_orspN_g > $W/curves_orspN_g.log 2>&1; echo "cg rc=$? $(date +%T)"
EXPDS_MATCH_SCOPE=site $L $PY repo/scripts/family_curves.py --fam $W/fam_orspN --gt $G --clusters work/family_clusters.json --out $W/curves_orspN_s > $W/curves_orspN_s.log 2>&1; echo "cs rc=$? $(date +%T)"
$L $PY work4/paired.py --pols keep --boot 1000 --base orspN=$W/curves_orspN_g --b orspR=/workspace/koz3_shared/bot-6/work/orsp2/curves_orspR_g --b orspv1=work9/curves_orsp_g --out $W/paired_N_vs.json > $W/paired_N_vs.log 2>&1; echo "p1 rc=$? $(date +%T)"
$L $PY work4/paired.py --pols keep --boot 1000 --base A=work/curves --b orspN=$W/curves_orspN_g --out $W/paired_N_A.json > $W/paired_N_A.log 2>&1; echo "p2 rc=$? $(date +%T)"
$L $PY work4/paired.py --pols keep --boot 1000 --base A=work4/curves_site --b orspN=$W/curves_orspN_s --out $W/paired_N_As.json > $W/paired_N_As.log 2>&1; echo "p3 rc=$? $(date +%T)"
echo done
