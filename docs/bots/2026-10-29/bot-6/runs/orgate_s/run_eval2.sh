#!/bin/bash
# bot-6 ORGATE_S, продолжение (11:0x): после lofo global — lofo site и seed0–4 только с G,Gext_S_rule (G_noLAS/Gext_S — диагностика, отброшены ради времени под нагрузкой).
cd /workspace/koz3_shared/bot-6/work/orgate_s
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 GDAL_NUM_THREADS=2 TMPDIR=/workspace/koz3_shared/bot-6/work/pytest_tmp PYTHONDONTWRITEBYTECODE=1
while pgrep -f "gate_eval_s.py --layout lofo --scope global" >/dev/null; do sleep 20; done
EMB=/workspace/koz3_shared/bot-6/work/orgate_s/emb_dinov2s_reg_112.npz
PY="prlimit --as=2000000000 nice -n 10 /workspace/rerank/venv/bin/python"
run(){ echo "$1 $2 $(date +%T)"; $PY gate_eval_s.py --layout $1 --scope $2 --variants G,Gext_S_rule --ext $EMB --pca 16 > res/$1_$2.log 2>&1; }
run lofo site
for L in seed0 seed1 seed2 seed3 seed4; do run $L global; done
for L in seed0 seed1 seed2 seed3 seed4; do run $L site; done
echo "done $(date +%T)"
