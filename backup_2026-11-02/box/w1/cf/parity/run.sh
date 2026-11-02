#!/bin/bash
cd /workspace/koz3/wt/w1cfd
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
PY=/workspace/koz3/w1/venv/bin/python; C=/workspace/koz3/w1/cf/ds/cands_cf.csv; REF=/workspace/koz3/w1/cf/box_out; O=/workspace/koz3/w1/cf/parity
L=/workspace/koz3/w2/las; G=/workspace/koz3/w2/gaudit/ds
run() { s=$1; las=$2; utm=$3; echo "== $s $(date +%H:%M:%S)"; nice -n 10 $PY tools/cloudfeat_parity.py --cands $C --ref $REF --site $s --las "$las" --utm "$utm" --json-out $O/p_$s.json 2> $O/time_$s.txt; echo "exit $?"; }
for d in $L/016_* $L/018_* $L/180_* $L/182_* $L/036_*; do s=$(basename $d | cut -d_ -f1); run $s "$(ls $d/01_*/*.las)" $d/UTM.json; done
for s in 015 021 186 190; do run $s $G/$s.las $G/${s}_UTM.json; done
echo ALLDONE
