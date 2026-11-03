#!/bin/bash
# W1-COMBO gate (PREREG docs/PREREG_W1_COMBO.md): frozen nested prep → prod replay COMBO → score vs base (+ info vs C1F, FP1).
set -euo pipefail
K=/workspace/koz3; W=$K/w1/satq/combo; PY=$K/w4venv/bin/python; cd "$(dirname "$0")"; mkdir -p $W
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
disk() { local a; a=$(df --output=avail -B1G /workspace | tail -1 | tr -d ' '); [ "$a" -ge 3 ] || { echo "BLOCKER: free disk ${a} GB < 3 GB"; exit 3; }; }
disk; [ -f $W/prep.pkl ] || nice -n 10 $PY combo_prep.py > $W/combo_prep.log 2>&1
disk; [ -s $W/out_COMBO.geojson ] || nice -n 10 $PY combo_replay.py COMBO $W/out_COMBO.geojson > $W/s2_COMBO.log 2>&1
disk; nice -n 10 $PY combo_score.py > $W/s3.log 2>&1
grep -E "delta|PASS|vs" $W/s3.log
