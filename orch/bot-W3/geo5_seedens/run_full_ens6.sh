#!/bin/bash
# D: full-train (all regions, no exclude) ens6 members for E1_G1 recipe (63lv/400r maggr finds; NOT L). n40r8full already exists (A). nice10, 3thr.
set -u; W=/workspace/koz3/w4geo/work; cd $W; PY="nice -n 10 $W/../venv/bin/python"; O=/workspace/koz3/orch/bot-W3/geo5_seedens
log(){ echo "$(date '+%H:%M:%S') [full] $*" | tee -a $O/run.log; }
for m in n40r6 n40r10 aug; do [ -f models_${m}full/ALL_s0/finds.txt ] || $PY sweep/retrain.py fit --cache feat_$m --out models_${m}full/ALL_s0 > $O/fit_full_$m.log 2>&1; log "fit $m rc=$?"; done
DR='Терраса_3_MagGr_(серый|цвет)\.tif$'
for g in 1 3; do m=gw${g}d; [ -f models_${m}full/ALL_s0/finds.txt ] || $PY sweep/fit_w.py --cache feat_n40r8 --out models_${m}full/ALL_s0 --gnz-w $g --drop-raster-re "$DR" > $O/fit_full_$m.log 2>&1; log "fit $m rc=$?"; done
log DONE
