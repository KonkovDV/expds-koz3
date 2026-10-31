#!/bin/bash
# разбивка R (Rrepro) по группам на k10; результат сохраняется как res/res_k10_Rgrp.json
cd /workspace/koz3_shared/bot-6/work/orgate; export OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 TMPDIR=/workspace/koz3_shared/bot-6/work/pytest_tmp
while pgrep -f run2.sh >/dev/null; do sleep 20; done
cp res/res_k10.json res/res_k10_grp.json
prlimit --as=2000000000 nice -n 10 /workspace/rerank/venv/bin/python gate_eval.py --layout k10 --variants Rrepro > res/k10_Rgrp.log 2>&1
cp res/res_k10.json res/res_k10_Rgrp.json; cp res/res_k10_grp.json res/res_k10.json; echo "done $(date +%T)"
