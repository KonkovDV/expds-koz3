cd /workspace/koz3/orch/bot-W3; P=/workspace/koz3/venv/bin/python
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
nice -n 10 $P code/make_joint.py split_arch variants/J || exit 1
V=""; for k in MRP ORG C5Bj S0 S1; do V="$V --variant $k=variants/J/$k.jsonl@variants/J/$k.drop"; done
nice -n 10 $P code/w3_gate.py --split-dir split_arch --gt /workspace/koz3/w5data/gt_train_v2.geojson --clusters runs/base_arch.json $V --pair S1:S0 --pair MRP:A0 --out runs/gate_J.json
echo JGATE_EXIT $?
