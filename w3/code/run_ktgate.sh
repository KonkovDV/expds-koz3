cd /workspace/koz3/orch/bot-W3; P=/workspace/koz3/venv/bin/python
nice -n 10 $P code/make_kt.py split_arch /workspace/koz3/ds/kcrops/kurgan_subclass_crops.npz /workspace/koz3/cpu/kurgan/work/kurgan_subclass_oof.csv variants/KT || exit 1
V=""; for k in KT1 KT2 KT3; do V="$V --variant $k=variants/KT/$k.jsonl@variants/KT/$k.drop"; done
nice -n 10 $P code/w3_gate.py --split-dir split_arch --gt /workspace/koz3/w5data/gt_train_v2.geojson --clusters runs/base_arch.json $V --out runs/gate_KT.json
echo KTGATE_EXIT $?
