cd /workspace/koz3/orch/bot-W3; P=/workspace/koz3/venv/bin/python
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
nice -n 10 $P code/make_nst.py split_arch variants/NST || exit 1
nice -n 10 $P code/w3_nested.py --split-dir split_arch --gt /workspace/koz3/w5data/gt_train_v2.geojson --clusters runs/base_arch.json --vdir variants/NST --out runs/gate_NST.json
echo NST_EXIT $?
