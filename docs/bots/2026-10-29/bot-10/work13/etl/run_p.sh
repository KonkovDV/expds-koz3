#!/bin/bash
cd /workspace/koz3_shared/bot-10
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
ulimit -v 3000000
nice -n 10 /workspace/rerank/venv/bin/python work4/paired.py --base A1=work13/etl/c_A_1 --b S1=work13/etl/c_S_1 --b Ss6_1=work13/etl/c_Ss6_1 --b A0=work/curves --out work13/etl/paired_layer_vsA1.json > work13/etl/paired_layer_vsA1.log 2>&1; echo "p rc=$? $(date +%T)"
