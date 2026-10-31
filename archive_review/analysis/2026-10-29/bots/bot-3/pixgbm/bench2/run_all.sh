#!/bin/bash
cd /workspace/koz3_shared/bot-3/pixgbm/bench2
PY=../../venv/bin/python; R="201 204 208 212 216 217 200"; ALL=maggr,etgr,terfl
run(){ ./runw2.sh bench2.log bench2.rss $PY bench2.py "$@"; }
for r in $R; do run $r 1 lightgbm 1 $ALL bench2.jsonl cand/${r}_lgb_1.pkl; done
for r in $R; do run $r 1 lleaves 1 $ALL bench2.jsonl cand/${r}_ll_1.pkl; done
for r in $R; do run $r 2 lightgbm 1 $ALL bench2.jsonl cand/${r}_lgb_2.pkl; done
for r in $R; do run $r 2 lleaves 1 $ALL bench2.jsonl cand/${r}_ll_2.pkl; done
for r in $R; do run $r 2 lleaves 2 $ALL bench2.jsonl cand/${r}_ll_2p2.pkl; done
for r in $R; do run $r 2 treelite 1 $ALL bench2.jsonl cand/${r}_tl_2.pkl; done
for nt in 1 2; do for r in $R; do run $r $nt lleaves $nt etgr,terfl preset.jsonl; done; done
for r in $R; do ./runw2.sh exact.log exact.rss $PY exact.py $r exact.jsonl; done
echo ALLDONE >> bench2.log
