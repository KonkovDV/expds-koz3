#!/bin/bash
# neigh: site-кривые (fam_nb из fam0) + NB_exact global (PREREG_audit, задача 2)
cd /workspace/koz3_shared/bot-10
export EXPDS_METRIC=$PWD/repo/materials/compute_metrics.py OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
PY=/workspace/rerank/venv/bin/python; G=/workspace/rerank/gt_train_v2.geojson
ulimit -v 3000000
echo "start $(date +%T)"
(
nice -n 10 $PY work11/xform_fam.py neigh --src work11/fam0 --dst work11/fam_nb --taus work11/taus_insample_A.json > work11/fam_nb2.log 2>&1; echo "fam rc=$? $(date +%T)"
EXPDS_MATCH_SCOPE=site nice -n 10 $PY repo/scripts/family_curves.py --fam work11/fam_nb --gt $G --clusters work/family_clusters.json --out work11/curves_nb_s > work11/curves_nb_s.log 2>&1; echo "cs rc=$? $(date +%T)"
rm -rf work11/fam_nb
nice -n 10 $PY work4/paired.py --pols drop,keep --base A=work4/curves_site --b nb=work11/curves_nb_s --out work11/nb_site.json > work11/nb_site.log 2>&1; echo "ps rc=$? $(date +%T)"
) &
BP=$!
cd work11; nice -n 10 $PY ../work6/taus.py NB /workspace/koz3_shared/bot-10/work11/curves_nb_g > taus_NB.log 2>&1; echo "taus_nb rc=$? $(date +%T)"
wait $BP
nice -n 10 $PY nb_exact.py global ../work6/taus_A.json taus_NB.json nb_exact_global.pkl > nb_exact_global.log 2>&1; echo "ex rc=$? $(date +%T)"
nice -n 10 $PY stats11.py nb_exact_global.pkl nb_exact_global.json NB-A,TONB-TO,TO-A,TONB-A > nb_exact_stats.log 2>&1; echo "st rc=$? $(date +%T)"
echo done
