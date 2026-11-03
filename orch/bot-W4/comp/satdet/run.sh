#!/bin/bash
# W4 local copy of wt/coord-comp/tools/satdet/run_det.sh (PREREG W1-SATDET 843d807); only paths changed (see diff in gate md).
set -uo pipefail
S=/workspace/koz3/orch/bot-W4/comp/satdet; O=$S/out; PY=/workspace/koz3/w4venv/bin/python; cd $S
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export EXPDS_METRIC=/workspace/koz3/wt/w1/materials/compute_metrics.py
nice -n 10 $PY t1_prep.py --det /workspace/koz3/w1/satdet/pulled/oof_frcnn_spor.geojson > $O/t1_det.log 2>&1 || { echo PREPFAIL >> $O/t1_det.log; exit 1; }
for v in base P1 P2 D1 D2; do nice -n 10 $PY t2_replay.py $v $O/out_$v.geojson $O/prep_det.pkl > $O/s2_$v.log 2>&1 || { echo REPLAYFAIL >> $O/s2_$v.log; exit 1; }; rm -f $O/out_$v.geojson.snap.pkl; done
sha256sum $O/out_*.geojson > $O/sha.txt
nice -n 10 $PY t3_score.py base P1 P2 D1 D2 > $O/s3_all.log 2>&1
echo ALLDONE >> $O/s3_all.log
