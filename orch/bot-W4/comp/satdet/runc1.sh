#!/bin/bash
set -uo pipefail
S=/workspace/koz3/orch/bot-W4/comp/satdet; O=$S/outc1; PY=/workspace/koz3/w4venv/bin/python; cd $S; mkdir -p $O
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export EXPDS_METRIC=/workspace/koz3/wt/w1/materials/compute_metrics.py
nice -n 10 $PY c1det_prep.py > $O/prep.log 2>&1 || { echo PREPFAIL >> $O/prep.log; exit 1; }
cp /workspace/koz3/v4build/gate/out/out_C1.geojson $O/out_C1.geojson
for v in ID C1D1f C1D2f; do nice -n 10 $PY c1det_replay.py $v $O/out_$v.geojson > $O/replay_$v.log 2>&1 || { echo REPLAYFAIL >> $O/replay_$v.log; exit 1; }; rm -f $O/out_$v.geojson.snap.pkl; done
sha256sum $O/out_*.geojson > $O/sha.txt
nice -n 10 $PY c1det_score.py C1 ID C1D1f C1D2f > $O/score_vs_C1.log 2>&1
echo ALLDONE >> $O/score_vs_C1.log
