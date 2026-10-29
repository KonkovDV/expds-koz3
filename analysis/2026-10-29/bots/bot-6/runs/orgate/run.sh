#!/bin/bash
# bot-6 ORGATE: полный прогон (k10, 5 семян, lofo). ≤2 потока, nice, ≤2 ГБ.
cd /workspace/koz3_shared/bot-6/work/orgate; export OMP_NUM_THREADS=2 GDAL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 TMPDIR=/workspace/koz3_shared/bot-6/work/pytest_tmp
PY="prlimit --as=2000000000 nice -n 10 /workspace/rerank/venv/bin/python"
[ -f feats_gate.npz ] || $PY feats_gate.py > feats_gate.log 2>&1
for L in ${LAYOUTS:-k10 seed0 seed1 seed2 seed3 seed4 lofo}; do $PY gate_eval.py --layout $L ${VARS:+--variants $VARS} ${EXT:+--ext $EXT} > res/$L${EXT:+_ext}.log 2>&1; done
echo done
