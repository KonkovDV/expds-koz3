#!/bin/bash
# C2 ConvNeXt (предварительно, фолды 0–3): R, RC, X → кривые global → S27-сабсет и полный → paired к A. fam_* удаляются сразу после кривых.
cd /workspace/koz3_shared/bot-10
export EXPDS_METRIC=$PWD/repo/materials/compute_metrics.py OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
PY=/workspace/rerank/venv/bin/python; G=/workspace/rerank/gt_train_v2.geojson
ulimit -v 3000000
while kill -0 854410 2>/dev/null; do sleep 20; done   # ждём run11a (лимит 2 потока)
echo "start $(date +%T)"
cur() { EXPDS_MATCH_SCOPE=global nice -n 10 $PY repo/scripts/family_curves.py --fam work11/fam_$1 --gt $G --clusters work/family_clusters.json --out work11/curves_$1_g > work11/curves_$1_g.log 2>&1; echo "$1 cg rc=$? $(date +%T)"; }
nice -n 10 $PY work11/cnx_fam.py R > work11/fam_cnxR.log 2>&1; echo "R fam rc=$? $(date +%T)"
cur cnxR
nice -n 10 $PY work11/xform_calib.py --src work11/fam_cnxR --dst work11/fam_cnxRC --mode mix --groups li_cnx > work11/fam_cnxRC.log 2>&1; echo "RC fam rc=$? $(date +%T)"
cur cnxRC
rm -rf work11/fam_cnxRC work11/fam_cnxR
nice -n 10 $PY work11/cnx_fam.py X > work11/fam_cnxX.log 2>&1; echo "X fam rc=$? $(date +%T)"
cur cnxX
rm -rf work11/fam_cnxX
cd work11; for v in cnxR cnxRC cnxX; do $PY cnx_subset.py curves_${v}_g curves_${v}_s27; done; cd ..
nice -n 10 $PY work4/paired.py --pols drop,keep --base A=work11/curves_A_s27 --b R=work11/curves_cnxR_s27 --b RC=work11/curves_cnxRC_s27 --b X=work11/curves_cnxX_s27 --out work11/cnx_s27.json > work11/cnx_s27.log 2>&1; echo "paired s27 rc=$? $(date +%T)"
nice -n 10 $PY work4/paired.py --pols drop,keep --base A=work/curves --b R=work11/curves_cnxR_g --b RC=work11/curves_cnxRC_g --b X=work11/curves_cnxX_g --out work11/cnx_full.json > work11/cnx_full.log 2>&1; echo "paired full rc=$? $(date +%T)"
echo done
cd /workspace/koz3_shared/bot-10/work11 && OMP_NUM_THREADS=1 nice -n 10 /workspace/rerank/venv/bin/python cnx_diag.py > cnx_diag.log 2>&1; echo "diag rc=$? $(date +%T)"
