#!/bin/bash
# bot-6 ORGATE_S: эмбеддинги DINOv2-S/reg@112 для 30 688 кандидатов bot-4 (2 потока, nice 10, ≤2 ГБ)
cd /workspace/koz3_shared/bot-6/work/orgate_s
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 GDAL_NUM_THREADS=2 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TMPDIR=/workspace/koz3_shared/bot-6/work/pytest_tmp PYTHONDONTWRITEBYTECODE=1
prlimit --as=2000000000 nice -n 10 /workspace/koz3_shared/bot-5/venv_det/bin/python embed_s.py > embed_s.log 2>&1
echo "done rc=$? $(date +%T)"
