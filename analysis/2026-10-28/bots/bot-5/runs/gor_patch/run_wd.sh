#!/bin/bash
# запуск с ограничениями COMMON.md: 2 потока, nice 10, сторож RSS 2000 МБ (убивает процесс), лог максимума RSS
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 HF_HUB_OFFLINE=1
LOG=$1; shift
nice -n 10 "$@" > "$LOG" 2>&1 & P=$!
MAX=0
while kill -0 $P 2>/dev/null; do
  R=$(ps -o rss= -p $P 2>/dev/null | tr -d ' '); R=${R:-0}
  [ "$R" -gt "$MAX" ] && MAX=$R
  if [ "$R" -gt 2048000 ]; then kill $P; echo "WATCHDOG: killed at RSS ${R} KB" >> "$LOG"; fi
  sleep 2
done
wait $P; E=$?
echo "watchdog max RSS MB $((MAX/1024)) exit $E" >> "$LOG"
