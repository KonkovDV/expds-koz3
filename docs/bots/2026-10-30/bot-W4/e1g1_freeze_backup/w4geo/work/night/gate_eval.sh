#!/bin/bash
# usage: gate_eval.sh TAG NAME...  (cstar/C_CetO3_Sr8G_noTerFl_L_E1_<NAME>.geojson -> gate raw+_ht, honest pce -> cstar/R_<TAG>.json)
set -u; TAG=$1; shift; W=/workspace/koz3/w4geo/work; cd $W; PY="nice -n 10 $W/../venv/bin/python"; export EXPDS_METRIC=/workspace/koz3/wt/w4-pix/materials/compute_metrics.py
L=(); for n in "$@"; do bash sweep/gate_cstar.sh CetO3_Sr8G_noTerFl_L_E1_$n & L+=(cstar/C_CetO3_Sr8G_noTerFl_L_E1_$n.geojson); done
cd /workspace/koz3/wt/w4-pix; $PY scripts/pix_clean_eval.py --gt /workspace/koz3/w5data/gt_train_v2.geojson --a $W/A18.geojson --c $(printf "$W/%s " "${L[@]}") --out $W/cstar/R_$TAG.json > $W/cstar/R_$TAG.log 2>&1
wait; echo "$(date '+%H:%M:%S') gate_eval $TAG done" >> $W/night/pipe.log
