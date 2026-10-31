#!/bin/bash
# bot-6 ORGATE_S: после эмбеддингов — замер времени, затем gate_eval_s.py (k10, lofo, seed0–4; global и site). ≤2 потока, nice 10, ≤2 ГБ.
cd /workspace/koz3_shared/bot-6/work/orgate_s
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 GDAL_NUM_THREADS=2 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TMPDIR=/workspace/koz3_shared/bot-6/work/pytest_tmp PYTHONDONTWRITEBYTECODE=1
while pgrep -f "embed_s.py" >/dev/null; do sleep 30; done
EMB=${EMB:-/workspace/koz3_shared/bot-6/work/orgate_s/emb_dinov2s_reg_112.npz}
[ -f $EMB ] || { echo "нет $EMB"; exit 1; }
PY="prlimit --as=2000000000 nice -n 10 /workspace/rerank/venv/bin/python"
[ -f bench_s.json ] || { echo "bench $(date +%T)"; prlimit --as=2000000000 nice -n 10 /workspace/koz3_shared/bot-5/venv_det/bin/python bench_s.py > bench_s.log 2>&1; }
for L in ${LAYOUTS:-k10 lofo seed0 seed1 seed2 seed3 seed4}; do
  V=G,G_noLAS,Gext_S_rule,Gext_S
  for SC in ${SCOPES:-global site}; do echo "$L $SC $(date +%T)"; $PY gate_eval_s.py --layout $L --scope $SC --variants $V --ext $EMB --pca 16 > res/${L}_${SC}.log 2>&1; done
done
echo "done $(date +%T)"
