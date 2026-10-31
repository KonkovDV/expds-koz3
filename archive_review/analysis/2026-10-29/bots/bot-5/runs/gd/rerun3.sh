#!/bin/bash
# bot-5: повтор ON после восстановления симлинка inference/lib/vendor в e2e/on (cp -r из репо его снёс)
export OMP_NUM_THREADS=2
PY=/workspace/koz3_shared/bot-7/venv/bin/python
W=/workspace/koz3_shared/bot-5/work/gd
cd $W/e2e
echo "== on2 $(date +%T)"
for reg in 084 081; do
  $PY $W/watch.py out/on2/$reg.log -- taskset -c 0,1 $PY on/inference/main.py in/$reg out/on2/$reg.geojson; tail -n 1 out/on2/$reg.log
  grep -E "gor_dense|GOR_DENSE" out/on2/$reg.log | tail -n 3
done
cmp out/on2/084.geojson out/off2/084.geojson && echo "084 ON==OFF"
$PY - <<'P'
import json
for r in ("084","081"):
    fs=json.load(open(f"/workspace/koz3_shared/bot-5/work/gd/e2e/out/on2/{r}.geojson"))["features"]
    print(r, len(fs), [(f["properties"].get("class_name") or f["properties"].get("class"), f["properties"].get("markup_type"), round(f["properties"].get("conf",0),5)) for f in fs][:6])
P
echo "RERUN3_DONE $(date +%T)"
