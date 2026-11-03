#!/bin/bash
cd /workspace/koz3/orch/bot-W4/comp/c1fp
PY=/workspace/koz3/w4geo/venv/bin/python; O=/workspace/koz3/orch/bot-W4/comp/c1fp/out
export EXPDS_METRIC=/workspace/koz3/wt/w1/materials/compute_metrics.py
nice -n 19 $PY c1fp_prep.py C1FP > prep.log 2>&1 || { echo PREPFAIL >> prep.log; exit 1; }
nice -n 19 $PY c1fp_replay.py C1FP $O/out_C1FP.geojson > replay_C1FP.log 2>&1 || { echo REPLAYFAIL >> replay_C1FP.log; exit 1; }
rm -f $O/out_C1FP.geojson.snap.pkl
nice -n 19 $PY c1fp_score.py base C1FP > score_vs_base.log 2>&1
nice -n 19 $PY c1fp_score.py C1 C1FP > score_vs_C1.log 2>&1
echo ALLDONE >> score_vs_base.log
