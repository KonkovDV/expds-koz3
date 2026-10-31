#!/bin/bash
cd /workspace/koz3_shared/bot-10; export OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1; ulimit -v 3000000
while kill -0 133957 2>/dev/null; do sleep 10; done
R="nice -n 10 /workspace/rerank/venv/bin/python work4/rerank/nested_lofo5.py --threads 1 --micro work4/rerank/pairs_arch_micro.pkl --scope site"
$R --layout k10 --variants full,noRG2 > work4/rerank/k10_site.log 2>&1
for s in 0 1 2 3 4; do $R --layout seed$s --variants full > work4/rerank/seed${s}_site.log 2>&1; done
$R --layout lofo --variants full > work4/rerank/lofo_site.log 2>&1
echo done
