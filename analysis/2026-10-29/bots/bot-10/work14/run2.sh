#!/bin/bash
cd /workspace/koz3_shared/bot-10/work14 || exit 1
export OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 LAYS=fam10,LOFO,seed0,seed1,seed2,seed3,seed4,seed5,seed6,seed7,seed8,seed9
nice -n 10 prlimit --as=3000000000 /workspace/rerank/venv/bin/python pkg_gor2.py > pkg_gor2.log 2>&1; echo "rc=$?" >> pkg_gor2.log
