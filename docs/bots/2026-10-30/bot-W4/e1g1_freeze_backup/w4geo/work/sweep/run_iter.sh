#!/bin/bash
# usage: run_iter.sh NAME NEG_RATIO NEG_MIN POS_R  (sample -> pipe.sh)
set -u; N=$1; cd /workspace/koz3/w4geo/work
nice -n 10 /workspace/koz3/w4geo/venv/bin/python sweep/retrain.py sample --cache feat_$N --neg-ratio $2 --neg-min $3 --pos-r $4 > sweep/sample_$N.log 2>&1
echo "$(date '+%H:%M:%S') [$N] sample rc=$? neg=$2/$3 pos_r=$4" >> sweep/pipe.log
sweep/pipe.sh $N feat_$N
