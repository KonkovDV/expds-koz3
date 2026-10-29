#!/bin/bash
# runw.sh <log> <rsslog> cmd... — nice 10, сторож RSS 1.95 ГБ, пик RSS в rsslog
log=$1; rl=$2; shift 2
export PYTHONPATH=/workspace/koz3_shared/bot-3/ehimen/pylib
nice -n 10 "$@" >> "$log" 2>&1 & pid=$!; peak=0
while kill -0 $pid 2>/dev/null; do r=$(ps -o rss= -p $pid | tr -d ' '); r=${r:-0}; [ "$r" -gt "$peak" ] && peak=$r
  [ "$r" -gt 1950000 ] && { echo "KILL rss=$r" >> "$log"; kill $pid; }; sleep 1; done
wait $pid; echo "$* peak_rss_kB=$peak exit=$?" >> "$rl"
