#!/bin/bash
# W1-FRZ gate (PREREG docs/PREREG_W1_FRZ.md 1532475): prep (gate-side nested, frozen non-sat τ) → prod replays → score. nice 10, <=4 thr.
set -euo pipefail
K=/workspace/koz3; W=$K/w1/satq/frz; PY=$K/w4venv/bin/python; cd "$(dirname "$0")"; mkdir -p $W
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
[ -f $W/prep.pkl ] || nice -n 10 $PY frz_prep.py > $W/frz_prep.log 2>&1
for v in base C1F A1F P2F; do nice -n 10 $PY frz_replay.py $v $W/out_$v.geojson > $W/s2_$v.log 2>&1; done
nice -n 10 $PY frz_score.py base C1F A1F P2F > $W/s3_all.log 2>&1
grep -E "delta|PASS" $W/s3_all.log
