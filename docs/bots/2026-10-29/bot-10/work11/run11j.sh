#!/bin/bash
# Доп. 5: FR1h (приор кадра без городищ)
cd /workspace/koz3_shared/bot-10
export EXPDS_METRIC=$PWD/repo/materials/compute_metrics.py OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
PY=/workspace/rerank/venv/bin/python; G=/workspace/rerank/gt_train_v2.geojson
ulimit -v 3000000
echo "start $(date +%T)"
# ждём окончания run11i (RAM)
while pgrep -f "run11i.sh" >/dev/null; do sleep 15; done; echo "run11i gone $(date +%T)"
nice -n 10 $PY work11/xform_frame.py --src work11/fam0 --dst work11/fam_fr1h --classes kurgany_povrezhdennye kurgany_tselye arkhitektury object_poly fortifikatsii finds_points > work11/fam_fr1h.log 2>&1; echo "fam rc=$? $(date +%T)"
EXPDS_MATCH_SCOPE=global nice -n 10 $PY repo/scripts/family_curves.py --fam work11/fam_fr1h --gt $G --clusters work/family_clusters.json --out work11/curves_fr1h_g > work11/curves_fr1h_g.log 2>&1; echo "cg rc=$? $(date +%T)"
EXPDS_MATCH_SCOPE=site nice -n 10 $PY repo/scripts/family_curves.py --fam work11/fam_fr1h --gt $G --clusters work/family_clusters.json --out work11/curves_fr1h_s > work11/curves_fr1h_s.log 2>&1; echo "cs rc=$? $(date +%T)"
nice -n 10 $PY work4/paired.py --pols drop,keep --base A=work/curves --b fr1h=work11/curves_fr1h_g --out work11/fr1h_global.json > work11/fr1h_global.log 2>&1; echo "pg rc=$? $(date +%T)"
nice -n 10 $PY work11/new34_breakdown.py work/curves work11/curves_fr1h_g work11/fr1h_new34.json > work11/fr1h_new34.log 2>&1; echo "n34 rc=$? $(date +%T)"
nice -n 10 $PY work4/paired.py --pols drop,keep --base A=work4/curves_site --b fr1h=work11/curves_fr1h_s --out work11/fr1h_site.json > work11/fr1h_site.log 2>&1; echo "ps rc=$? $(date +%T)"
cd work11; nice -n 10 $PY ../work6/taus.py FR1h /workspace/koz3_shared/bot-10/work11/curves_fr1h_g > taus_FR1h.log 2>&1; echo "taus rc=$? $(date +%T)"
echo done
