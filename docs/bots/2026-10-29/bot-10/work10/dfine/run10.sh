#!/bin/bash
# bot-10 work10: независимый перемер D-FINE (OOF bot-9 oof_dfine_5f.geojson, вариант C) — global, site, поверх пакета S+stack6.
cd /workspace/koz3_shared/bot-10/work10/dfine
B=/workspace/koz3_shared/bot-10
export EXPDS_METRIC=$B/repo/materials/compute_metrics.py OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
PY=/workspace/rerank/venv/bin/python; G=/workspace/rerank/gt_train_v2.geojson; CL=$B/work/family_clusters.json
ulimit -v 3000000
echo "start $(date +%T) df $(df -h / | awk 'NR==2{print $4}')"
nice -n 10 $PY prep10.py gt > gt.log 2>&1; echo "gt rc=$? $(date +%T)"
nice -n 10 $PY prep10.py fam base > fam_base.log 2>&1; echo "fam base rc=$? $(date +%T)"
nice -n 10 $PY prep10.py fam C > fam_C.log 2>&1; echo "fam C rc=$? $(date +%T) df $(df -h / | awk 'NR==2{print $4}')"
for sc in g s; do S=global; [ $sc = s ] && S=site
  for v in base C; do EXPDS_MATCH_SCOPE=$S nice -n 10 $PY $B/repo/scripts/family_curves.py --fam fam_$v --gt $G --clusters $CL --out curves_${v}_$sc > curves_${v}_$sc.log 2>&1; echo "c ${v}_$sc rc=$? $(date +%T)"; done
done
nice -n 10 $PY asm10.py > asm10.log 2>&1; echo "asm rc=$? $(date +%T)"
cd $B
nice -n 10 $PY work4/paired.py --pols drop,keep --base A=work/curves --b C=work10/dfine/c5_Cg --out work10/dfine/paired_g.json > work10/dfine/paired_g.log 2>&1; echo "pg rc=$? $(date +%T)"
nice -n 10 $PY work4/paired.py --pols drop,keep --base A=work4/curves_site --b C=work10/dfine/c5_Cs --out work10/dfine/paired_s.json > work10/dfine/paired_s.log 2>&1; echo "ps rc=$? $(date +%T)"
nice -n 10 $PY work4/paired.py --pols keep --base PKG=work7/curves_g_Ss6 --b PKC=work10/dfine/PKC_g --out work10/dfine/paired_pkg.json > work10/dfine/paired_pkg.log 2>&1; echo "pp rc=$? $(date +%T)"
echo done
