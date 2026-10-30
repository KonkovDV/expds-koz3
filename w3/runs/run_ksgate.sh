cd /workspace/koz3/orch/bot-W3; P=/workspace/koz3/venv/bin/python; mkdir -p variants/KS
nice -n 10 $P code/make_k.py split_arch work_ks_avg_oof.csv variants/KS p_povr KS || exit 1
nice -n 10 $P code/make_kt.py split_arch /workspace/koz3/ds/kcrops/kurgan_subclass_crops.npz work_ks_avg_oof.csv variants/KS KST || exit 1
V=""; for k in KS1 KS2 KS3 KS4 KS5 KST; do f=variants/KS/$k.jsonl; [ -f variants/KS/$k.drop ] && f="$f@variants/KS/$k.drop"; V="$V --variant $k=$f"; done
nice -n 10 $P code/w3_gate.py --split-dir split_arch --gt /workspace/koz3/w5data/gt_train_v2.geojson --clusters runs/base_arch.json $V --out runs/gate_KS.json
echo KSGATE_EXIT $?
