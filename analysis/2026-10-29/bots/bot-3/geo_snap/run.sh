#!/bin/bash
# bash run.sh <name> <args predict_peaks.py>; ulimit -v 2,2 ГБ + сторож RSS 1,95 ГБ (вся группа процессов), лог <name>.rss
set -u
name=$1; shift
W=/workspace/koz3_shared/bot-3/geo_snap; cd $W
export PYTHONPATH=/workspace/koz3_shared/bot-3/ehimen/pylib OMP_NUM_THREADS=2
( ulimit -v 2200000; exec setsid nice -n 10 ../venv/bin/python predict_peaks.py "$@" ) > $W/$name.log 2>&1 &
pid=$!; peak=0
while kill -0 $pid 2>/dev/null; do
  r=$(ps -o rss= --ppid $pid -p $pid 2>/dev/null | awk '{s+=$1} END{print s+0}')
  [ "$r" -gt "$peak" ] && peak=$r
  if [ "$r" -gt 1950000 ]; then echo "KILL rss=$r kB" >> $W/$name.rss; kill -- -$pid 2>/dev/null || kill $pid; fi
  sleep 2
done
wait $pid; ec=$?
echo "exit=$ec peak_rss_kB=$peak" >> $W/$name.rss
