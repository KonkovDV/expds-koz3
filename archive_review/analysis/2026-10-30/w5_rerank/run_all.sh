#!/usr/bin/env bash
# W5 full run. Needs $W5_DATA/stack_arch.geojson (sha b412e299…, from PC C:\Sloi\rerank) + gt_train_v2.geojson (sha d4bc1a3b…).
set -euo pipefail; cd "$(dirname "$0")"; PY=${PY:-/workspace/koz3/venv/bin/python}; N="nice -n 10"
export W5_DATA=${W5_DATA:-/workspace/koz3/w5data}
$N $PY prep.py; $N $PY check.py || true          # check.py: fast.curves == build_curves, honest == 0.314615
$N $PY an1.py; $N $PY feats.py; $N $PY feats2.py
$N $PY w5_policies.py --variants V0,V1,V2,V3,V4 --boot 2000 --out w5_policies.json
$N $PY nested2.py --tag w5all --label first --threads 3
$N $PY w5_policies.py --variants V0,V5 --rerank $W5_DATA/work/nested_w5all.npz --out w5_rerank.json
$N $PY export_cands.py
