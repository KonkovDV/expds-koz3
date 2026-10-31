#!/bin/bash
# bot-10 work12: независимая проверка ORGATE (bot-6). Пишет только в bot-10/work12.
cd /workspace/koz3_shared/bot-10/work12; export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 GDAL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
PY="prlimit --as=2000000000 nice -n 10 /workspace/rerank/venv/bin/python"
echo "feats $(date +%T)"; $PY feats_gate.py > feats_gate.log 2>&1
$PY -c "
import numpy as np;a=np.load('feats_gate.npz');b=np.load('/workspace/koz3_shared/bot-6/work/orgate/feats_gate.npz')
print('same idx',(a['stack_idx']==b['stack_idx']).all(),'same names',list(a['names'])==list(b['names']),'max|dX|',float(np.nanmax(np.abs(a['X']-b['X']))),'nan eq',(np.isnan(a['X'])==np.isnan(b['X'])).all())" > feats_cmp.log 2>&1
for job in "k10 global G_noLAS,G,Rrepro" "k10 site G_noLAS,Rrepro" "lofo global G_noLAS" "lofo site G_noLAS" \
           "seed0 global G_noLAS" "seed1 global G_noLAS" "seed2 global G_noLAS" "seed3 global G_noLAS" "seed4 global G_noLAS" \
           "seed0 site G_noLAS" "seed1 site G_noLAS" "seed2 site G_noLAS" "seed3 site G_noLAS" "seed4 site G_noLAS"; do
  set -- $job; echo "$1 $2 $3 $(date +%T)"
  $PY gate_eval12.py --layout $1 --scope $2 --variants $3 > res/$1_$2.log 2>&1 || echo "FAIL $1 $2 rc=$?"
done
echo "done $(date +%T)"
