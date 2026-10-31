#!/bin/bash
# смоук D-FINE на CPU коробки: 2 потока, nice 10, сторож RSS 2000 МБ
cd /workspace/koz3_shared/bot-5/repo
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 HF_HOME=/workspace/koz3_shared/bot-5/tmp/hf TOKENIZERS_PARALLELISM=false
ulimit -v ${VLIM:-2500000}
nice -n 10 ../venv_det/bin/python scripts/det_train_hf.py --data /workspace/koz3_shared/bot-5/tmp/smoke_coco --out /workspace/koz3_shared/bot-5/tmp/$1 --model ${MODEL:-dfine-l} --size ${SIZE:-256} --threads 2 $2 &
P=$!; MAX=0
while kill -0 $P 2>/dev/null; do R=$(ps -o rss= -p $P 2>/dev/null | tr -d ' '); [ -n "$R" ] && [ "$R" -gt "$MAX" ] && MAX=$R
  if [ -n "$R" ] && [ "$R" -gt 2048000 ]; then echo "WATCHDOG: RSS ${R}KB > 2000MB, killing"; kill $P; fi; sleep 1; done
wait $P; echo "exit $? watchdog_max_rss_MB $((MAX/1024))"
