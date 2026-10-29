#!/bin/bash
cd /workspace/koz3_shared/bot-7/repo
PY=../../bot-8/venv/bin/python; W6=../work6; Z=$1; T=$2; IN=$3
L=$W6/check_$T.log
../work5/run.sh $L $PY scripts/check_submission.py $Z --python $PY --cpus 0 --lock requirements.lock --keep ${IN:+--input $IN}
W=$(grep -o "/tmp/b8smoke_[A-Za-z0-9_]*" $L | tail -1)
mkdir -p $W6/out/$T
if [ -n "$W" ]; then cp $W/run.log $W6/check_${T}_run.log; cp $W/out/result.geojson $W6/out/$T/ 2>/dev/null; chmod -R u+w $W; rm -rf $W; fi
echo "$T done: $(grep ИТОГ $L)"
