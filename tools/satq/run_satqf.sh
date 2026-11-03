#!/bin/bash
# W1-SATQ frozen gate (PREREG docs/PREREG_W1_SATQ.md + docs/PREREG_W1_SATQ_AMEND1.md): 1) frozen gate-side nested prep
# 2) prod replays on v3_on b447649  3) score (m=6, Bonf 0.98333, Q1 0.99). nice 10, <=4 threads. Stops if free disk < 3 GB.
set -euo pipefail
K=/workspace/koz3; W=$K/w1/satq/frzq; PY=$K/w4venv/bin/python; cd "$(dirname "$0")"; mkdir -p $W
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
disk() { local a; a=$(df --output=avail -B1G /workspace | tail -1 | tr -d ' '); [ "$a" -ge 3 ] || { echo "BLOCKER: free disk ${a} GB < 3 GB"; exit 3; }; }
disk; [ -f $W/prep.pkl ] || nice -n 10 $PY satqf_prep.py > $W/satqf_prep.log 2>&1
for v in base Q1a Q1b Q1c FP1 FP2 Q2; do disk; [ -s $W/out_$v.geojson ] || nice -n 10 $PY satqf_replay.py $v $W/out_$v.geojson > $W/s2_$v.log 2>&1; done
disk; nice -n 10 $PY satqf_score.py base Q1a Q1b Q1c FP1 FP2 Q2 > $W/s3_all.log 2>&1
grep -E "delta|PASS" $W/s3_all.log
