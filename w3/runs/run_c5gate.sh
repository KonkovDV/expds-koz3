cd /workspace/koz3/orch/bot-W3; P=/workspace/koz3/venv/bin/python
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
nice -n 10 $P code/make_c5.py split_arch variants/C5 || exit 1
V=""; for k in C5A C5B C5C; do V="$V --variant $k=variants/C5/$k.jsonl@variants/C5/$k.drop"; done
nice -n 10 $P code/w3_gate.py --split-dir split_arch --gt /workspace/koz3/w5data/gt_train_v2.geojson --clusters runs/base_arch.json $V --out runs/gate_C5.json
echo C5GATE_EXIT $?
