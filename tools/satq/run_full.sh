#!/bin/bash
# W1-FRZ full-data production config (docs/W1_C1F_HANDOFF.md): full-data selection → prod replays (same τ in all folds; in-sample) → score.
set -euo pipefail
K=/workspace/koz3; W=$K/w1/satq/frz; PY=$K/w4venv/bin/python; cd "$(dirname "$0")"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
[ -f $W/full.pkl ] || nice -n 10 $PY full_prep.py > $W/full_prep.log 2>&1
for v in B_full C1F_full P2F_full; do nice -n 10 $PY frz_replay.py $v $W/out_$v.geojson $W/full.pkl > $W/s2_$v.log 2>&1; done
nice -n 10 $PY frz_score.py B_full C1F_full P2F_full > $W/s3_full.log 2>&1
grep -E "delta|PASS" $W/s3_full.log || true
