#!/bin/bash
# bot-6 ORGATE часть 2 (после lofo): разбивка по группам + G_noLAS (k10), --scope site (k10, seed0–4), Gext с эмбеддингами bot-4 (k10, seed0–4; опционально).
cd /workspace/koz3_shared/bot-6/work/orgate; export OMP_NUM_THREADS=2 GDAL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 TMPDIR=/workspace/koz3_shared/bot-6/work/pytest_tmp
PY="prlimit --as=2000000000 nice -n 10 /workspace/rerank/venv/bin/python"
while pgrep -f "gate_eval.py --layout lofo" >/dev/null; do sleep 30; done
mkdir -p res_v1 && cp -n res/* res_v1/ 2>/dev/null
echo "grp $(date +%T)"; $PY gate_eval.py --layout k10 --variants G,G_noLAS,RULE > res/k10_grp.log 2>&1
for L in ${SITE_LAYOUTS:-k10 seed0 seed1 seed2 seed3 seed4}; do echo "site $L $(date +%T)"; $PY gate_eval.py --layout $L --scope site --variants G > res/${L}_site.log 2>&1; done
EMB=/workspace/koz3_shared/bot-4/work/cropverif/emb_dinov2b_reg_112.npz
if [ -f $EMB ]; then for L in ${EXT_LAYOUTS:-k10 seed0 seed1 seed2 seed3 seed4}; do echo "ext $L $(date +%T)"; $PY gate_eval.py --layout $L --ext $EMB --pca 16 --variants Gext > res/${L}_ext.log 2>&1; done; fi
echo "done $(date +%T)"
