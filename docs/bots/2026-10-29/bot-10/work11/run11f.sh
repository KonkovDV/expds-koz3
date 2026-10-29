#!/bin/bash
# C2b: ConvNeXt c3-in фолд 4 (PREREG_audit Доп. 4)
cd /workspace/koz3_shared/bot-10
export EXPDS_METRIC=$PWD/repo/materials/compute_metrics.py OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
PY=/workspace/rerank/venv/bin/python; G=/workspace/rerank/gt_train_v2.geojson
ulimit -v 3000000
IN=/workspace/kaggle_out/koz3-cnx-li-c3-f4/oof_cnx_c3-in_f4.geojson
echo "start $(date +%T)"
cur() { EXPDS_MATCH_SCOPE=global nice -n 10 $PY repo/scripts/family_curves.py --fam work11/fam_$1 --gt $G --clusters work/family_clusters.json --out work11/curves_$1_g > work11/curves_$1_g.log 2>&1; echo "$1 cg rc=$? $(date +%T)"; }
nice -n 10 $PY work11/cnx_fam2.py cnxRin R 4 $IN > work11/fam_cnxRin.log 2>&1; echo "Rin fam rc=$? $(date +%T)"
cur cnxRin
nice -n 10 $PY work11/xform_calib.py --src work11/fam_cnxRin --dst work11/fam_cnxRCin --mode mix --groups li_cnx > work11/fam_cnxRCin.log 2>&1; echo "RCin fam rc=$? $(date +%T)"
cur cnxRCin
rm -rf work11/fam_cnxRCin work11/fam_cnxRin
nice -n 10 $PY work11/cnx_fam2.py cnxXin X 4 $IN > work11/fam_cnxXin.log 2>&1; echo "Xin fam rc=$? $(date +%T)"
cur cnxXin
rm -rf work11/fam_cnxXin
cd work11; $PY cnx_subset2.py 4 ../work/curves curves_A_s4; for v in cnxRin cnxRCin cnxXin; do $PY cnx_subset2.py 4 curves_${v}_g curves_${v}_s4; done; cd ..
nice -n 10 $PY work4/paired.py --pols drop,keep --base A=work11/curves_A_s4 --b Rin=work11/curves_cnxRin_s4 --b RCin=work11/curves_cnxRCin_s4 --b Xin=work11/curves_cnxXin_s4 --out work11/cnxin_s4.json > work11/cnxin_s4.log 2>&1; echo "paired s4 rc=$? $(date +%T)"
cd work11 && CNX_FOLDS=4 CNX_OUT=cnxin_diag.json CNX_V='{"A":"curves_A_s4","R":"curves_cnxRin_s4","RC":"curves_cnxRCin_s4","X":"curves_cnxXin_s4"}' nice -n 10 $PY cnx_diag.py > cnxin_diag.log 2>&1; echo "diag rc=$? $(date +%T)"; cd ..
nice -n 10 $PY work4/paired.py --pols drop,keep --base A=work/curves --b Rin=work11/curves_cnxRin_g --b RCin=work11/curves_cnxRCin_g --b Xin=work11/curves_cnxXin_g --out work11/cnxin_full.json > work11/cnxin_full.log 2>&1; echo "paired full rc=$? $(date +%T)"
echo done
