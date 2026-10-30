#!/bin/bash
cd /workspace/koz3/w4clean
E=/workspace/koz3/w4geo/eval
OMP_NUM_THREADS=3 nice -n 10 /workspace/koz3/w4geo/bin/peakrun.sh $E/A.peak /workspace/koz3/w4venv/bin/python scripts/geo_predict.py \
  --root /workspace/koz3/w4geo/train --config configs/v3_on.toml --set geophys.pixgbm.enabled=false \
  --out $E/A.geojson
echo ADONE
