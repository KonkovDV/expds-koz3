#!/bin/bash
# bot-7: последовательные check_submission (--cpus 0, --lock, --keep) — сохраняем run.log и result.geojson
cd /workspace/koz3_shared/bot-7/repo
PY=../../bot-8/venv/bin/python
one() { # sha tag input
  local Z=../work5/dist/submission_bot7cand_v2_$1.zip L=../work5/check_$1_$2.log
  [ "$2" = synth ] && IN="" || IN="--input $3"
  ../work5/run.sh $L $PY scripts/check_submission.py $Z --python $PY --cpus 0 --lock requirements.lock --keep $IN
  W=$(grep -o "/tmp/b8smoke_[A-Za-z0-9_]*" $L | tail -1)
  mkdir -p ../work5/out/$1_$2
  if [ -n "$W" ]; then cp $W/run.log ../work5/check_$1_$2_run.log; cp $W/out/result.geojson ../work5/out/$1_$2/; chmod -R u+w $W; rm -rf $W; fi
  echo "$1 $2 done: $(grep ИТОГ $L)"
}








S=eb268d6
EXPDS_PIXGBM_THREADS=1 ../work5/run.sh ../work5/pytest_$S.log $PY -m pytest -q -p no:cacheprovider -k "not coco_roundtrip" tests; echo "pytest: $(tail -1 ../work5/pytest_$S.log)"
../work5/run.sh ../work5/build_$S.log $PY scripts/build_submission.py --out ../work5/dist/submission_bot7cand_v2_$S.zip; echo "build: $(tail -1 ../work5/build_$S.log)"
one $S synth
one $S 210 ../work5/in/210
one $S 214 ../work5/in/214
one $S 217 ../work5/in/217
echo ALLDONE
