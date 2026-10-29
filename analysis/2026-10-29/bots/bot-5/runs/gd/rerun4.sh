#!/bin/bash
# bot-5: ветка bot5-gor-dense-v4 (HEAD на f3526cf). Флаг ВЫКЛ. vs чистый v4 f3526cf (base4) на 084/081/210/217; флаг ВКЛ. (on4) на 084/081.
export OMP_NUM_THREADS=2
PY=/workspace/koz3_shared/bot-7/venv/bin/python
W=/workspace/koz3_shared/bot-5/work/gd; cd $W/e2e
echo "== v4 $(date +%T)"; mkdir -p out/base4 out/off4 out/on4
for reg in 084 081 210 217; do
  IN=in/$reg; [ -d $IN ] || IN=/workspace/koz3_shared/bot-7/work5/in/$reg
  for r in base4 off4; do $PY $W/watch.py out/$r/$reg.log -- taskset -c 0,1 $PY $r/inference/main.py $IN out/$r/$reg.geojson; tail -n 1 out/$r/$reg.log; done
  cmp out/base4/$reg.geojson out/off4/$reg.geojson && echo "$reg IDENTICAL $(md5sum < out/off4/$reg.geojson | cut -c1-32) $(stat -c%s out/off4/$reg.geojson)B"
done
for reg in 084 081; do
  $PY $W/watch.py out/on4/$reg.log -- taskset -c 0,1 $PY on4/inference/main.py in/$reg out/on4/$reg.geojson; tail -n 1 out/on4/$reg.log
  grep -E "GOR_DENSE|UNAVAILABLE|gor_dense: сырых|по слоям" out/on4/$reg.log | tail -n 3
done
cmp out/on4/084.geojson out/off4/084.geojson && echo "084 ON==OFF"
$PY - <<'P'
import json
for r in ("084","081"):
    fs=json.load(open(f"/workspace/koz3_shared/bot-5/work/gd/e2e/out/on4/{r}.geojson"))["features"]
    print(r, len(fs), [{k:v for k,v in f["properties"].items() if k in ("class_name","markup_type","confidence","conf","original_crs","region_name")} for f in fs][:4])
P
echo "RERUN4_DONE $(date +%T)"
