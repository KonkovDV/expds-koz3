#!/bin/bash
# usage: run.sh <layouts> <tag>
cd /workspace/koz3_shared/bot-4/work/morph
OMP_NUM_THREADS=2 nice -n 10 /workspace/rerank/venv/bin/python eval_morph.py --layouts "$1" --variants C,M > res/log_$2.txt 2>&1
echo "EXIT $? $(date '+%F %T')" >> res/log_$2.txt
