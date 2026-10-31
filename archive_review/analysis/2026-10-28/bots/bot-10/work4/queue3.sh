#!/bin/bash
cd /workspace/koz3_shared/bot-10; export OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 EXPDS_METRIC=$PWD/repo/materials/compute_metrics.py; ulimit -v 3000000
while kill -0 130878 2>/dev/null; do sleep 10; done
nice -n 10 /workspace/rerank/venv/bin/python work4/xcheck_bot8.py Гнёздово Боровичи Курганы Иссык+Каджы Миловидово+Осечки+Примаки Вышегжа Семибратнее Судьбищи > work4/xcheck_bot8.log 2>&1
echo done
