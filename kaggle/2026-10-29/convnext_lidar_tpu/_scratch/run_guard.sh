#!/bin/bash
# run_guard.sh <log> <rss_out> <cmd...>: nice -n 10, 2 потока, пик RSS дерева процессов (своя группа через setsid); убивает группу при > CAP_KB
log=$1; rss=$2; shift 2; CAP=${CAP_KB:-2000000}
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 RAYON_NUM_THREADS=2 TF_NUM_INTEROP_THREADS=1 TF_NUM_INTRAOP_THREADS=2
setsid nice -n 10 "$@" > "$log" 2>&1 & p=$!; peak=0
tree_rss() { local s=0; for q in $(pgrep -s $p 2>/dev/null || echo $p); do r=$(awk '/VmRSS/{print $2}' /proc/$q/status 2>/dev/null); s=$((s+${r:-0})); done; echo $s; }
while kill -0 $p 2>/dev/null; do r=$(tree_rss); [ "$r" -gt "$peak" ] && peak=$r
  if [ "$r" -gt "$CAP" ]; then echo "RSS $r КБ > $CAP — убито" >> "$log"; kill -9 -- -$p 2>/dev/null; fi; sleep 0.5; done
wait $p; rc=$?; echo "peak_rss_kb=$peak rc=$rc" > "$rss"; echo "exit=$rc" >> "$log"; exit $rc
