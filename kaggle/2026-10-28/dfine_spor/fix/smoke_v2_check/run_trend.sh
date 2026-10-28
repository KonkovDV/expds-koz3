#!/bin/bash
cd /workspace/koz3_shared/bot-9/kaggle/dfine_spor/fix; O=$PWD/smoke_v2_check; L=$O/train256_f3.log
( set -a; . ~/.koz3s3/env; set +a; export AWS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 HF_HOME=$PWD/hf HF_HUB_OFFLINE=1 DFINE_OUT=$O/train256_f3 DFINE_CACHE=$PWD/s3cache DFINE_CHIPS=$O/chips256 DFINE_WORKERS=0 DFINE_TILE=256 DFINE_BS=2 CHK_TRAIN_ITERS=32 CHK_FOLD=3; exec nice -n 10 venv_t211/bin/python smoke_v2_check.py ) > $L 2>&1 & P=$!; PEAK=0
while kill -0 $P 2>/dev/null; do R=$(ps -o rss= -p $P | awk '{s+=$1}END{print s+0}'); [ $R -gt $PEAK ] && PEAK=$R; [ $R -gt 2048000 ] && { echo WATCHDOG >> $L; kill $P; }; sleep 3; done
wait $P; echo "rc=$? peak_kB=$PEAK" >> $L
