#!/bin/bash
# bot-7 v3: сборка (флаги выкл.) + новый check_submission (bot-8 7878087 + наши) на synth/210/214/217 и на zip v2 eb268d6
cd /workspace/koz3_shared/bot-7/repo
PY=../../bot-8/venv/bin/python; S=$1; W6=../work6
one() { # zip tag input
  local Z=$1 L=$W6/check_$2.log
  [ -z "$3" ] && IN="" || IN="--input $3"
  ../work5/run.sh $L $PY scripts/check_submission.py $Z --python $PY --cpus 0 --lock requirements.lock --keep $IN
  W=$(grep -o "/tmp/b8smoke_[A-Za-z0-9_]*" $L | tail -1)
  mkdir -p $W6/out/$2
  if [ -n "$W" ]; then cp $W/run.log $W6/check_$2_run.log; cp $W/out/result.geojson $W6/out/$2/ 2>/dev/null; chmod -R u+w $W; rm -rf $W; fi
  echo "$2 done: $(grep ИТОГ $L)"
}
mkdir -p $W6/dist
../work5/run.sh $W6/build_$S.log $PY scripts/build_submission.py --out $W6/dist/submission_bot7cand_v3off_$S.zip; echo "build: $(tail -1 $W6/build_$S.log)"
for t in synth 210 214 217; do
  [ $t = synth ] && IN="" || IN=../work5/in/$t
  one $W6/dist/submission_bot7cand_v3off_$S.zip v3off_${S}_$t $IN
  cmp $W6/out/v3off_${S}_$t/result.geojson ../work5/out/eb268d6_$t/result.geojson && echo "BYTE-IDENTICAL $t" || echo "DIFF $t"
done
for t in $V2; do
  [ $t = synth ] && IN="" || IN=../work5/in/$t
  one ../work5/dist/submission_bot7cand_v2_eb268d6.zip v2zip_newchk_$t $IN
done
echo ALLDONE
