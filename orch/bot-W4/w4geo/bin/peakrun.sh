#!/bin/bash
# peakrun.sh <peakfile> cmd... — запускает команду и пишет пиковый RSS (кБ) её дерева процессов
pf=$1; shift
"$@" & P=$!
mx=0
while kill -0 $P 2>/dev/null; do
  r=$(ps -o rss= --ppid $P -p $P 2>/dev/null | awk '{s+=$1} END{print s+0}')
  [ "$r" -gt "$mx" ] && mx=$r && echo $mx > $pf
  sleep 1
done
wait $P; rc=$?
echo "$mx rc=$rc" > $pf
exit $rc
