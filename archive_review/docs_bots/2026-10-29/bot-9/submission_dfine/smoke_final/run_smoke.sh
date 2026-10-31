#!/bin/bash
# CPU-смоук в рамках COMMON.md: 2 потока, nice 10, сторож RSS 1,95 ГБ (дерево процесса), HF офлайн.
cd /workspace/koz3_shared/bot-9/submission_dfine/smoke_final || exit 1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 GDAL_NUM_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
PY=/workspace/koz3_shared/bot-9/kaggle/dfine_spor/fix/venv_t211/bin/python
nice -n 10 $PY ../export_fp16.py /workspace/kaggle_out/koz3-dfine-spor-final/dfine_final /workspace/koz3_shared/bot-9/submission_dfine/smoke_final/dfine_final_fp16 > export.log 2>&1; echo "export rc=$?"
nice -n 10 $PY smoke.py /workspace/koz3_shared/bot-9/submission_dfine/smoke_final/dfine_final_fp16 > smoke.json 2> smoke.err & P=$!
peak=0
while kill -0 $P 2>/dev/null; do r=$(ps -o rss= -p $P | tr -d ' '); r=${r:-0}; [ $r -gt $peak ] && peak=$r
  if [ $r -gt 1996800 ]; then echo "RSS $r KB > 1.95 GB — kill"; kill $P; fi; sleep 1; done
wait $P; echo "smoke rc=$? peak_rss_KB=$peak"
