#!/bin/bash
# Запуск шага с лимитом памяти: ulimit -v 2,2 ГБ + сторож RSS (kill при ≥1,9 ГБ). Лог RSS — <name>.rss
# Использование: bash run.sh <name> <args scripts/pixgbm.py ...>
set -u
name=$1; shift
cd /workspace/koz3_shared/bot-3/repo
export PYTHONPATH=/workspace/koz3_shared/bot-3/ehimen/pylib OMP_NUM_THREADS=2
W=/workspace/koz3_shared/bot-3/pixgbm
( ulimit -v 2200000; exec nice -n 10 ../venv/bin/python scripts/pixgbm.py "$@" ) > $W/$name.log 2>&1 &
pid=$!
peak=0
while kill -0 $pid 2>/dev/null; do
  r=$(ps -o rss= -p $pid 2>/dev/null | tr -d ' '); r=${r:-0}
  [ "$r" -gt "$peak" ] && peak=$r
  if [ "$r" -gt 1950000 ]; then echo "KILL rss=$r kB" >> $W/$name.rss; kill $pid; fi
  sleep 2
done
wait $pid; ec=$?
echo "exit=$ec peak_rss_kB=$peak" >> $W/$name.rss
