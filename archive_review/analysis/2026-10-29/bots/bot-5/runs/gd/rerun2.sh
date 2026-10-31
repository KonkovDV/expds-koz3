#!/bin/bash
# bot-5: финальная проверка на коде 30c7169+ (фаза до gor_gate). Ждёт окончания rerun_all.sh (одна тяжёлая задача за раз).
export OMP_NUM_THREADS=2
PY=/workspace/koz3_shared/bot-7/venv/bin/python
W=/workspace/koz3_shared/bot-5/work/gd; R=/workspace/koz3_shared/bot-5/repo
while ! grep -q RERUN_DONE $W/rerun_all.log; do sleep 20; done
cd $W/e2e; echo "== repo $(git -C $R log --oneline -1)"
# ON-копия: код репо + конфиг репо, где [gor_dense] enabled=false заменён на включённый блок
for d in expds scripts inference; do rm -rf on/$d && cp -r $R/$d on/$d; done
$PY - <<'P'
from pathlib import Path
src = Path('/workspace/koz3_shared/bot-5/repo/configs/default.toml').read_text(encoding='utf-8')
old = "[gor_dense]\nenabled = false\n"; assert src.count(old) == 1
new = ('[gor_dense]\nenabled = true\nbackbone = "models/dinov2-small"\ngate_head = "models/gor_gate/gor_v3_head.npz"\n'
       'head = "models/gor_dense/gor_dense_head.npz"\nhead_md5 = "e35570e10d17edac32bea8d26fa0f5f4"\ncentre_s = 250.0\n\n'
       '[thresholds_by_markup.gor_dense]\ngorodishcha = 0.824066\n')
Path('/workspace/koz3_shared/bot-5/work/gd/e2e/on/configs/default.toml').write_text(src.replace(old, new), encoding='utf-8')
import tomllib; c = tomllib.loads(src.replace(old, new)); print("on cfg", c["gor_dense"], c["thresholds_by_markup"]["gor_dense"])
P
echo "== off2 $(date +%T)"; mkdir -p out/off2 out/on2
for reg in 084 081 210 217; do
  IN=in/$reg; [ -d $IN ] || IN=/workspace/koz3_shared/bot-7/work5/in/$reg
  $PY $W/watch.py out/off2/$reg.log -- taskset -c 0,1 $PY off/inference/main.py $IN out/off2/$reg.geojson; tail -n 1 out/off2/$reg.log
  cmp out/base/$reg.geojson out/off2/$reg.geojson && echo "$reg IDENTICAL $(md5sum < out/off2/$reg.geojson | cut -c1-32) $(stat -c%s out/off2/$reg.geojson)B"
done
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
echo "RERUN2_DONE $(date +%T)"
