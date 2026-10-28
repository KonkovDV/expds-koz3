#!/bin/bash
# usage: run_dryrun.sh <venv> <outdir> [VAR=val ...]  — CPU-дриран собранного smoke-ядра, 2 потока, nice 10, сторож RSS 2000 МБ.
set -u
H=/workspace/koz3_shared/bot-9/kaggle/dfine_spor; VENV=$1; OUT=$2; shift 2
mkdir -p "$OUT"
( set -a; . ~/.koz3s3/env; set +a
  export AWS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
    HF_HOME=$H/fix/hf HF_HUB_OFFLINE=1 DFINE_OUT=$OUT DFINE_CACHE=$H/fix/s3cache DFINE_CHIPS=$OUT/chips \
    DFINE_TILE=256 DFINE_MAX_WINDOWS=6 DFINE_BS=1 DFINE_ITERS=3 DFINE_WORKERS=0 DFINE_SMOKE_TRAIN_REGIONS=2 \
    DFINE_REGIONS=161_Курганы_32,051_Воронеж_2 "$@"
  exec nice -n 10 "$VENV/bin/python" ${SCRIPT:-$H/smoke/koz3-dfine-spor-smoke.py} --allow-cpu
) > "$OUT/dryrun.log" 2>&1 &
PID=$!; PEAK=0
while kill -0 $PID 2>/dev/null; do
  R=$( (ps -o rss= -p $PID; ps -o rss= --ppid $PID) 2>/dev/null | awk '{s+=$1} END{print s+0}')
  [ "$R" -gt "$PEAK" ] && PEAK=$R
  if [ "$R" -gt 2048000 ]; then echo "WATCHDOG: RSS ${R} kB > 2000 MB — kill" >> "$OUT/dryrun.log"; pkill -P $PID; kill $PID; fi
  sleep 2
done
wait $PID; RC=$?; echo "rc=$RC watchdog_peak_kB=$PEAK" >> "$OUT/dryrun.log"; echo "rc=$RC"
