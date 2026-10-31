#!/bin/bash
# ждёт окончания morph k10 (PID 1076424), затем вариант R (PREREG_MORPH_RING) на k10; повтор с чекпойнта после OOM
cd /workspace/koz3_shared/bot-4/work/morph
while kill -0 1076424 2>/dev/null; do sleep 20; done
for try in 1 2 3 4 5; do
  [ -f res/res_k10_ring.json ] && break
  OMP_NUM_THREADS=2 nice -n 10 /workspace/rerank/venv/bin/python eval_morph.py --layouts k10 --variants R --tag _ring > res/log_k10_ring.r$try.txt 2>&1
  echo "EXIT $? $(date '+%F %T')" >> res/log_k10_ring.r$try.txt
  sleep 30
done
