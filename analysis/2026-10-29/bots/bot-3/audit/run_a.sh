#!/bin/bash
# bash run_a.sh <name> <python args...>: ulimit -v 2,2 ГБ, сторож RSS 1,95 ГБ, nice 10, OMP 2; лог audit/out/<name>.log/.rss
set -u
name=$1; shift
A=/workspace/koz3_shared/bot-3/audit
cd $A/wt
export PYTHONPATH=/workspace/koz3_shared/bot-3/ehimen/pylib OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
( ulimit -v 2200000; exec nice -n 10 /workspace/koz3_shared/bot-3/venv/bin/python "$@" ) > $A/out/$name.log 2>&1 &
pid=$!; peak=0
while kill -0 $pid 2>/dev/null; do
  r=$(ps -o rss= -p $pid 2>/dev/null | tr -d ' '); r=${r:-0}
  [ "$r" -gt "$peak" ] && peak=$r
  if [ "$r" -gt 1950000 ]; then echo "KILL rss=$r kB" >> $A/out/$name.rss; kill $pid; fi
  sleep 2
done
wait $pid; ec=$?
echo "exit=$ec peak_rss_kB=$peak" >> $A/out/$name.rss
