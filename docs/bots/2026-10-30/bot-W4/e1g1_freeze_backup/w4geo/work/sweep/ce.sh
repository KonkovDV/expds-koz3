#!/bin/bash
# fixed compose+eval tail of pipe.sh (abs venv path). usage: ce.sh NAME
set -u; N=$1; PY="nice -n 10 /workspace/koz3/w4geo/venv/bin/python"
W=/workspace/koz3/w4geo/work; T=/workspace/koz3/w4geo/train
log(){ echo "$(date '+%H:%M:%S') [$N] $*" | tee -a $W/sweep/pipe.log; }
cd /workspace/koz3/wt/w4-pix; export EXPDS_METRIC=$PWD/materials/compute_metrics.py
mkdir -p $W/c_$N
for tg in $(cat $W/sweep/tags_${N}_small.txt); do
 for U in S U; do ex=(); [ $U = U ] && ex=(--set 'geophys.pixgbm.union_name_re=".*"')
  $PY scripts/pix_clean_oof.py compose --root $T --config configs/v3_on.toml --cache $W/cache --tag $tg --set 'geophys.pixgbm.impl="clean"' --set geophys.pixgbm.replace=true --set 'geophys.pixgbm.maps=["maggr"]' --set 'geophys.pixgbm.classes=["finds_points"]' "${ex[@]}" --out $W/c_$N/C_${U}_$tg.geojson > $W/c_$N/compose_${U}_$tg.log 2>&1 &
 done; done; wait; log compose
$PY scripts/pix_clean_eval.py --gt /workspace/koz3/w5data/gt_train_v2.geojson --a $W/A18.geojson --c $W/c_$N/C_*.geojson --out $W/c_$N/R.json > $W/c_$N/eval.log 2>&1; log "eval rc=$?"
