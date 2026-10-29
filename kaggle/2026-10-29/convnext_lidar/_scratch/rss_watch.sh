#!/bin/bash
# пик RSS дерева процессов (КБ) каждые 2 с → $2
pid=$1; out=$2; peak=0
while kill -0 $pid 2>/dev/null; do
  s=$(ps -o rss= --ppid $pid -p $pid 2>/dev/null | awk '{s+=$1} END{print s+0}')
  for c in $(pgrep -P $pid); do s=$((s+$(ps -o rss= --ppid $c 2>/dev/null | awk '{s+=$1} END{print s+0}'))); done
  [ "$s" -gt "$peak" ] && peak=$s && echo "peak_rss_kb=$peak" > $out
  if [ "$s" -gt 1950000 ]; then echo "KILL rss=$s" >> $out; pkill -TERM -P $pid; kill -TERM $pid; fi
  sleep 2
done
