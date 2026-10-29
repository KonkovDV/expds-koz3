#!/bin/bash
# bot-4: ждёт окончания эмбеддинга (PID 802173), затем строгая оценка по PREREG_CROPVERIF.md
cd /workspace/koz3_shared/bot-4/work/cropverif
while kill -0 802173 2>/dev/null; do sleep 60; done
[ -f emb_dinov2b_reg_112.npz ] || { echo "нет эмбеддингов, стоп"; exit 1; }
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
PY=/workspace/rerank/venv/bin/python
echo "start k10 $(date)"
nice -n 10 $PY evalrr.py --layouts k10 --variants X,E,X:kur,E:kur,X:arch,E:arch,X:fort,E:fort > eval_k10.log 2>&1
echo "start lofo+seeds $(date)"
nice -n 10 $PY evalrr.py --layouts lofo,seed0,seed1,seed2,seed3,seed4 --variants X,E > eval_p3.log 2>&1
echo "done $(date)"
