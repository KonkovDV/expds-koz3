#!/bin/bash
# usage: run_gw.sh NAME GNZ_W DROP_RE   (fit_w over feat_n40r8 -> pipe.sh tail)
set -u; N=$1; GW=$2; DR=$3; cd /workspace/koz3/w4geo/work; PY="nice -n 10 /workspace/koz3/w4geo/venv/bin/python"
for f in "F1 200" "F2 201 202 203 204 205 206 207" "F3 208 209 210 215" "F4 211 212 216" "F5 213 214 217"; do set -- $f; k=$1; shift
  [ -f models_$N/${k}_s0/finds.txt ] || $PY sweep/fit_w.py --cache feat_n40r8 --out models_$N/${k}_s0 --exclude $@ --gnz-w $GW --drop-raster-re "$DR" >> sweep/fit_$N.log 2>&1; done
echo "$(date '+%H:%M:%S') [$N] fit_w done gw=$GW drop=$DR" >> sweep/pipe.log
sweep/pipe.sh $N feat_n40r8
