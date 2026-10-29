#!/bin/bash
# neigh site: fam_nb из fam0 (код 6e26062 из git archive → work11/wt_audit), site-кривые, парный site
cd /workspace/koz3_shared/bot-10
export EXPDS_METRIC=$PWD/repo/materials/compute_metrics.py OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
PY=/workspace/rerank/venv/bin/python; G=/workspace/rerank/gt_train_v2.geojson
ulimit -v 3000000
nice -n 10 $PY work11/xform_fam.py neigh --src work11/fam0 --dst work11/fam_nb --taus work11/taus_insample_A.json > work11/fam_nb2.log 2>&1; R=$?; echo "fam rc=$R $(date +%T)"; [ $R -eq 0 ] || exit 1
EXPDS_MATCH_SCOPE=site nice -n 10 $PY repo/scripts/family_curves.py --fam work11/fam_nb --gt $G --clusters work/family_clusters.json --out work11/curves_nb_s > work11/curves_nb_s.log 2>&1; R=$?; echo "cs rc=$R $(date +%T)"; [ $R -eq 0 ] || exit 1
rm -rf work11/fam_nb
nice -n 10 $PY work4/paired.py --pols drop,keep --base A=work4/curves_site --b nb=work11/curves_nb_s --out work11/nb_site.json > work11/nb_site.log 2>&1; echo "ps rc=$? $(date +%T)"
echo done
