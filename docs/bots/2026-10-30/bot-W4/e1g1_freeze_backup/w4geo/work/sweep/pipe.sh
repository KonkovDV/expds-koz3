#!/bin/bash
# usage: pipe.sh NAME FEATCACHE [extra retrain fit args...]
set -u; N=$1; FC=$2; shift 2; X="$@"
cd /workspace/koz3/w4geo/work; PY="nice -n 10 /workspace/koz3/w4geo/venv/bin/python"
log(){ echo "$(date '+%H:%M:%S') [$N] $*" | tee -a sweep/pipe.log; }
for f in "F1 200" "F2 201 202 203 204 205 206 207" "F3 208 209 210 215" "F4 211 212 216" "F5 213 214 217"; do set -- $f; k=$1; shift
  [ -f models_$N/${k}_s0/finds.txt ] || $PY sweep/retrain.py fit --cache $FC --out models_$N/${k}_s0 --exclude $@ $X >> sweep/fit_$N.log 2>&1; done; log fit
$PY sweep/finds_sweep.py dump --models models_$N --probs probs_$N --threads 3 > sweep/dump_$N.log 2>&1; log dump
$PY sweep/finds_sweep.py decode --probs probs_$N --prefix ${N}_ --grid small --calib raw max > sweep/decode_$N.log 2>&1; log decode
cd /workspace/koz3/wt/w4-pix; W=/workspace/koz3/w4geo/work; T=/workspace/koz3/w4geo/train; export EXPDS_METRIC=$PWD/materials/compute_metrics.py
mkdir -p $W/c_$N
for tg in $(cat $W/sweep/tags_${N}_small.txt); do
 for U in S U; do ex=(); [ $U = U ] && ex=(--set 'geophys.pixgbm.union_name_re=".*"')
  $PY scripts/pix_clean_oof.py compose --root $T --config configs/v3_on.toml --cache $W/cache --tag $tg --set 'geophys.pixgbm.impl="clean"' --set geophys.pixgbm.replace=true --set 'geophys.pixgbm.maps=["maggr"]' --set 'geophys.pixgbm.classes=["finds_points"]' "${ex[@]}" --out $W/c_$N/C_${U}_$tg.geojson > /dev/null 2>&1 &
 done; done; wait; log compose
$PY scripts/pix_clean_eval.py --gt /workspace/koz3/w5data/gt_train_v2.geojson --a $W/A18.geojson --c $W/c_$N/C_*.geojson --out $W/c_$N/R.json > $W/c_$N/eval.log 2>&1; log "eval rc=$?"
