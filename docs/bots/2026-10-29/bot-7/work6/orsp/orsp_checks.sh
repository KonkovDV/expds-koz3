#!/bin/bash
# bot-7: сквозные проверки ORSP (a)–(e) на участке 103 (SpOr есть). Запускать ПОСЛЕ приёмки bot-10; флаг в сдаче не включается.
#   bash orsp_checks.sh <zip выкл.> <povr sp_focal> <тег> [skip_off]   (≤2 ГБ, 1 поток, nice; распаковка во /tmp, удаляется в конце)
set -e
W=/workspace/koz3_shared/bot-7/work6; PY=/workspace/koz3_shared/bot-8/venv/bin/python; Z=$1; POVR=$2; TAG=$3; SKIP=$4; T=$(mktemp -d /tmp/b7orsp_XXXX)
$PY $W/orsp/make_orsp.py $Z $T/on.zip $POVR
mkdir -p $T/off $T/on; (cd $T/off && unzip -q $Z); (cd $T/on && unzip -q $T/on.zip); rm $T/on.zip
for v in off on; do
  [ $v = off ] && N=off || N=on_$TAG
  [ $v = off ] && [ -n "$SKIP" ] && [ -f $W/orsp/out_off.geojson ] && continue
  taskset -c 0,1 $W/../work5/run.sh $W/orsp/run_$N.log $PY $W/orsp/run_hooked.py $T/$v $W/in/103 $W/orsp/out_$N.geojson $W/orsp/dump_$N.json
  tail -1 $W/orsp/run_$N.log
done
$PY $W/e2e/e2e_score.py --metric /workspace/koz3_shared/bot-7/repo/materials/compute_metrics.py --gt $W/e2e/gt_103.geojson \
  --a off=$W/orsp/out_off.geojson --b on=$W/orsp/out_on_$TAG.geojson --out $W/orsp/e2e_orsp_$TAG.json
$PY $W/orsp/orsp_report.py $W/orsp/dump_off.json $W/orsp/out_off.geojson $W/orsp/dump_on_$TAG.json $W/orsp/out_on_$TAG.geojson $W/orsp/e2e_orsp_$TAG.json | tee $W/orsp/orsp_checks_$TAG.txt
chmod -R u+w $T; rm -rf $T
