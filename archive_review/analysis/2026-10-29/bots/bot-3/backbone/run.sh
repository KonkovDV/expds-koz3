#!/bin/bash
# Шаг с лимитом: ulimit -v 4 ГБ (torch резервирует виртуальную память) + сторож RSS (сумма с детьми, kill при ≥1,9 ГБ). Лог RSS — <name>.rss
# bash run.sh <name> <args scripts/pixgbm_bb.py ...>
set -u
name=$1; shift
cd /workspace/koz3_shared/bot-3/repo
W=/workspace/koz3_shared/bot-3/backbone
export PYTHONPATH=$W/pylib:/workspace/koz3_shared/bot-3/ehimen/pylib OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 TMPDIR=/tmp/bb3
mkdir -p /tmp/bb3
( exec nice -n 10 ../venv/bin/python scripts/pixgbm_bb.py "$@" ) > $W/$name.log 2>&1 &
pid=$!
peak=0
while kill -0 $pid 2>/dev/null; do
  r=$(ps -o rss= --ppid $pid -p $pid 2>/dev/null | awk '{s+=$1} END {print s+0}')
  [ "$r" -gt "$peak" ] && peak=$r
  if [ "$r" -gt 1950000 ]; then echo "KILL rss=$r kB" >> $W/$name.rss; kill $pid; fi
  sleep 2
done
wait $pid; ec=$?
echo "exit=$ec peak_rss_kB=$peak $(date '+%F %T')" >> $W/$name.rss
