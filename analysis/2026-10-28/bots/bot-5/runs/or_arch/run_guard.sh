#!/bin/bash
# $1 model $2 tif $3 region $4 out ; RSS watchdog 1800 MB, 2 threads, nice 10
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 GDAL_CACHEMAX=256
ulimit -v ${VLIM:-4000000}
nice -n 10 /workspace/koz3_shared/bot-5/venv_det/bin/python /workspace/koz3_shared/bot-5/work/or_arch/run_or.py "$@" &
P=$!; MX=0
while kill -0 $P 2>/dev/null; do R=$(( $(awk '/VmRSS/{print $2}' /proc/$P/status 2>/dev/null || echo 0) / 1024 )); [ $R -gt $MX ] && MX=$R
  if [ $R -gt 1800 ]; then echo "WATCHDOG kill rss=$R"; kill -9 $P; fi; sleep 1; done
wait $P; echo "exit $? watchdog_max_rss_MB $MX"
