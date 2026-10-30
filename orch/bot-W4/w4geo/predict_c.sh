#!/bin/bash
# C: OOF-предсказания pix_clean по фолдам, сид $1
set -e
s=$1
cd /workspace/koz3/w4clean
PY=/workspace/koz3/w4venv/bin/python
M=/workspace/koz3/w4geo/models
E=/workspace/koz3/w4geo/eval
T=/workspace/koz3/w4geo/train
declare -A F=( [F1]="^200_" [F2]="^20[1-7]_" [F3]="^(208|209|210|215)_" [F4]="^(211|212|216)_" [F5]="^(213|214|217)_" )
for k in F1 F2 F3 F4 F5; do
  regs=$(ls $T | grep -E "${F[$k]}")
  echo "== $k s$s: $regs"
  OMP_NUM_THREADS=3 $PY scripts/geo_predict.py --root $T --config configs/v3_on.toml \
    --set geophys.pixgbm.enabled=true --set 'geophys.pixgbm.impl="clean"' \
    --set "geophys.pixgbm.model_dir=\"$M/${k}_s$s\"" --set geophys.pixgbm.replace=true \
    --set 'geophys.pixgbm.union_name_re=""' --set geophys.pixgbm.ground_metres=false \
    --set geophys.pixgbm.procs=3 --regions $regs --out $E/C_s${s}_$k.geojson
done
$PY - <<PYEOF
import json
fs=[json.load(open(f"$E/C_s${s}_{k}.geojson")) for k in ("F1","F2","F3","F4","F5")]
out={**{k:v for k,v in fs[0].items() if k!="features"},"features":[f for fc in fs for f in fc["features"]]}
json.dump(out,open("$E/C_s${s}.geojson","w"),ensure_ascii=False)
print("C_s$s", len(out["features"]))
PYEOF
