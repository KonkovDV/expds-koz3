cd /workspace/koz3/orch/bot-W3; P=/workspace/koz3/venv/bin/python
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
nice -n 10 $P code/gap.py --split-dir split_arch --gt /workspace/koz3/w5data/gt_train_v2.geojson --clusters runs/base_arch.json --variant variants/NST/m0_O --out runs/gap_breakdown --dump runs/gap_cands.pkl
echo GAP_EXIT $?
