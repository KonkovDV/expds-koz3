#!/bin/bash
set -e
cd /workspace/koz3/w4clean
PY=/workspace/koz3/w4venv/bin/python
C=/workspace/koz3/w4geo/cache_clean
M=/workspace/koz3/w4geo/models
declare -A F=( [F1]="200" [F2]="201 202 203 204 205 206 207" [F3]="208 209 210 215" [F4]="211 212 216" [F5]="213 214 217" )
[ -f $M/full_s0/meta.json ] || $PY scripts/pix_clean_train.py fit --cache $C --out $M/full_s0 --seed 0 --threads 3
for s in 0 1 2; do for k in F1 F2 F3 F4 F5; do
  [ -f $M/${k}_s$s/meta.json ] && continue; echo "== $k s$s"; $PY scripts/pix_clean_train.py fit --cache $C --out $M/${k}_s$s --seed $s --threads 3 --exclude ${F[$k]}
done; done
echo ALLFIT
