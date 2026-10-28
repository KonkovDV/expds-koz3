#!/bin/bash
# по одному процессу; 1 и 2 потока; сторож RSS 1,95 ГБ
cd /workspace/koz3_shared/bot-3/pixgbm/bench
export PYTHONPATH=/workspace/koz3_shared/bot-3/ehimen/pylib
for nt in 1 2; do for r in 201 204 208 212 216 217 200; do
  ( ulimit -v 2400000; OMP_NUM_THREADS=$nt exec nice -n 10 ../../venv/bin/python bench.py $r $nt bench.jsonl ) >> bench.log 2>&1 &
  pid=$!; peak=0
  while kill -0 $pid 2>/dev/null; do r2=$(ps -o rss= -p $pid | tr -d ' '); r2=${r2:-0}; [ "$r2" -gt "$peak" ] && peak=$r2
    [ "$r2" -gt 1950000 ] && { echo "KILL $r $nt" >> bench.log; kill $pid; }; sleep 1; done
  echo "$r nt=$nt peak_rss_kB=$peak" >> bench.rss
done; done
echo ALLDONE >> bench.log
