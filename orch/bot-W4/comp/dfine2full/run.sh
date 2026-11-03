#!/bin/bash
cd /workspace/koz3/orch/bot-W4/comp/dfine2full
PY=/workspace/koz3/w4geo/venv/bin/python; O=/workspace/koz3/orch/bot-W4/comp/dfine2full/out
export EXPDS_METRIC=/workspace/koz3/wt/w1/materials/compute_metrics.py
nice -n 19 $PY dfine2full_prep.py > prep.log 2>&1 || { echo PREPFAIL >> prep.log; exit 1; }
for v in ID E8C1 E8C1f; do
  nice -n 19 $PY dfine2full_replay.py $v $O/out_$v.geojson > replay_$v.log 2>&1 || { echo REPLAYFAIL >> replay_$v.log; exit 1; }
  rm -f $O/out_$v.geojson.snap.pkl
done
sha256sum $O/out_ID.geojson $O/out_C1.geojson $O/out_E8C1.geojson $O/out_E8C1f.geojson > sha.txt
nice -n 19 $PY dfine2full_score.py C1 ID E8C1 E8C1f > score_vs_C1.log 2>&1
nice -n 19 $PY dfine2full_score.py base E8C1 E8C1f > score_vs_base.log 2>&1
echo ALLDONE >> score_vs_C1.log
