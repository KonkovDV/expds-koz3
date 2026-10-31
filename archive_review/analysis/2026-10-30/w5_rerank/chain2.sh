#!/usr/bin/env bash
# after nested2 (w5all, w5cnx): V5/V5x gate vs A, then V6c (stack + fused cnx) prep + gate vs A.
cd "$(dirname "$0")"; PY=/workspace/koz3/venv/bin/python; N="nice -n 10"; B=/workspace/koz3/w5data; C=/workspace/koz3/w5data_cnx
while [ ! -f $B/work/nested_w5all.npz ] || [ ! -f $B/work/nested_w5cnx.npz ]; do sleep 60; done
W5_DATA=$B $N $PY -u w5_policies.py --variants V0,V5 --rerank $B/work/nested_w5all.npz --out w5_rerank.json
W5_DATA=$B $N $PY -u w5_policies.py --variants V0,V5 --rerank $B/work/nested_w5cnx.npz --out w5_rerank_cnx.json
W5_DATA=$B $N $PY -u gate_v6c.py --save base_counts.pkl
W5_DATA=$C W5_STACK=stack_arch_cnx.geojson $N $PY -u prep.py | tail -n 3
W5_DATA=$C $N $PY -u gate_v6c.py --vs $B/work/base_counts.pkl --out w6c_gate.json
echo CHAIN2DONE
