#!/bin/bash
# usage: watchdog.sh LIMIT_KB cmd...  — убивает процесс, если RSS > LIMIT_KB
LIM=$1; shift
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
nice -n 10 "$@" & P=$!; PEAK=0
while kill -0 $P 2>/dev/null; do
  R=$(ps -o rss= -p $P 2>/dev/null | tr -d ' '); R=${R:-0}
  [ "$R" -gt "$PEAK" ] && PEAK=$R
  if [ "$R" -gt "$LIM" ]; then echo "WATCHDOG: RSS $R kB > $LIM, kill" >&2; kill -9 $P; fi
  sleep 2
done
wait $P; RC=$?; echo "WATCHDOG: rc=$RC peak_rss_kB=$PEAK" >&2; exit $RC
