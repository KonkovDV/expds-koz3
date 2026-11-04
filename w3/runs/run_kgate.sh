#!/bin/bash
cd /workspace/koz3/orch/bot-W3
P=/workspace/koz3/venv/bin/python
nice -n 10 $P code/make_k.py split_arch /workspace/koz3/cpu/kurgan/work/kurgan_subclass_oof.csv variants/K p_povr K || exit 1
V=""
for k in K1 K2 K3 K4 K5; do f=variants/K/$k.jsonl; [ -f variants/K/$k.drop ] && f="$f@variants/K/$k.drop"; V="$V --variant $k=$f"; done
nice -n 10 $P code/w3_gate.py --split-dir split_arch --gt /workspace/koz3/w5data/gt_train_v2.geojson --clusters runs/base_arch.json $V --out runs/gate_K.json
echo KGATE_EXIT $?
