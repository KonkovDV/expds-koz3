#!/bin/bash
# runw2.sh <log> <rsslog> cmd... — nice 10, сторож суммарного RSS (процесс + дети) 1,95 ГБ
log=$1; rl=$2; shift 2
export PYTHONPATH=/workspace/koz3_shared/bot-3/ehimen/pylib
nice -n 10 "$@" >> "$log" 2>&1 & pid=$!; peak=0
while kill -0 $pid 2>/dev/null; do
  r=$(ps -o rss= --pid $pid --ppid $pid 2>/dev/null | awk '{s+=$1} END{print s+0}')
  [ "$r" -gt "$peak" ] && peak=$r
  [ "$r" -gt 1950000 ] && { echo "KILL rss=$r" >> "$log"; pkill -P $pid; kill $pid; }; sleep 1; done
wait $pid; ec=$?; echo "$* peak_rss_sum_kB=$peak exit=$ec" >> "$rl"
