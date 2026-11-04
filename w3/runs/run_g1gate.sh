cd /workspace/koz3/orch/bot-W3; P=/workspace/koz3/venv/bin/python
nice -n 10 $P code/make_g1.py split_arch variants/G1 || exit 1
V=""; for k in G1A G1B G1C; do V="$V --variant $k=variants/G1/$k.jsonl@variants/G1/$k.drop"; done
nice -n 10 $P code/w3_gate.py --split-dir split_arch --gt /workspace/koz3/w5data/gt_train_v2.geojson --clusters runs/base_arch.json $V --out runs/gate_G1.json
echo G1GATE_EXIT $?
