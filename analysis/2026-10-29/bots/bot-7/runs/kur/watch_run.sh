#!/bin/bash
# usage: watch_run.sh LOG cmd... ; kills child if RSS > 2000 MB
LOG=$1; shift
"$@" > "$LOG" 2>&1 & P=$!; MX=0
while kill -0 $P 2>/dev/null; do R=$(ps -o rss= -p $P 2>/dev/null | tr -d ' '); R=${R:-0}; [ $R -gt $MX ] && MX=$R
 if [ $R -gt 2048000 ]; then echo "WATCHDOG KILL rss=${R}KB" >> "$LOG"; kill $P; fi; sleep 2; done
wait $P; E=$?; echo "watchdog max RSS MB $((MX/1024)) exit $E" >> "$LOG"
