#!/bin/bash
cd /workspace/koz3/orch/bot-W4/work/c1f
PY=/workspace/koz3/w4venv/bin/python; O=/workspace/koz3/w4/c1f/out
while pgrep -f c1f_prep.py >/dev/null; do sleep 10; done
for v in base C1 P2; do nice -n 19 $PY c1f_replay.py $v $O/out_$v.geojson > replay_$v.log 2>&1; rm -f $O/out_$v.geojson.snap.pkl; done
nice -n 19 $PY c1f_score.py base C1 P2 > score.log 2>&1
echo ALLDONE >> score.log
