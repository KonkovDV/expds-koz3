#!/bin/bash
cd /workspace/koz3/orch/bot-W4/work/c1fx
PY=/workspace/koz3/w4venv/bin/python; O=/workspace/koz3/w4/c1fx/out
nice -n 19 $PY c1fx_prep.py C1 GA FP1 > prep.log 2>&1 || { echo PREPFAIL >> prep.log; exit 1; }
for v in base C1 GA FP1; do nice -n 19 $PY c1fx_replay.py $v $O/out_$v.geojson > replay_$v.log 2>&1; rm -f $O/out_$v.geojson.snap.pkl; done
nice -n 19 $PY c1fx_score.py base C1 GA FP1 > score.log 2>&1
echo ALLDONE >> score.log
