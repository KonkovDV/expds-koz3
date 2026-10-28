#!/bin/bash
# пробы ресурсов каждые 5 с, пока жив run.sh (pid $1): RSS hillfort (МБ), load, MemAvailable
while kill -0 $1 2>/dev/null; do
  r=$(ps -eo rss=,args= | grep '[h]illfort_oof.py --root roots' | awk '{s+=$1} END{print int(s/1024)}')
  echo "$(date +%T) rss_mb=${r:-0} load=$(cut -d' ' -f1 /proc/loadavg) avail_mb=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)"
  sleep 5
done; echo END
