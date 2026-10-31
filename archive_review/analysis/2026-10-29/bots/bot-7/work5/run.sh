#!/bin/bash
# bot-7: запуск по COMMON.md — 1 поток, nice 10, сторож RSS 1900 МБ, лог max RSS.  run.sh LOG PY script.py ...
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
LOG=$1; shift
nice -n 10 "$@" > "$LOG" 2>&1 & P=$!
MAX=0
while kill -0 $P 2>/dev/null; do
  R=$(ps -o rss= -p $P 2>/dev/null | tr -d ' '); R=${R:-0}
  [ "$R" -gt "$MAX" ] && MAX=$R
  if [ "$R" -gt 1945600 ]; then kill $P; echo "WATCHDOG: killed at RSS ${R} KB" >> "$LOG"; fi
  sleep 1
done
wait $P; E=$?
echo "watchdog max RSS MB $((MAX/1024)) exit $E" >> "$LOG"
