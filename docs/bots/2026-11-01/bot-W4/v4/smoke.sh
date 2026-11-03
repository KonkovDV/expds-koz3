#!/bin/bash
S=/workspace/koz3/w4geo/smoke190; X=$1; OUT=$2
while pgrep -f w5cf_infer_parity > /dev/null; do sleep 15; done
cd $X; export PYTHONPATH=/workspace/koz3/w1/cf/ds/laspy-2.7.0-py3-none-any.whl:/workspace/koz3/w4geo/kaggle/ds-w4-wheels/segmentation_models_pytorch-0.5.0-py3-none-any.whl:/workspace/koz3/w4geo/pydeps_smoke; PY=${PY:-/workspace/koz3/w1/venv/bin/python}
mkdir -p $S/$OUT
$PY $S/rss.py $PY inference/main.py $S/in $S/$OUT/result.geojson > $S/$OUT.log 2> $S/$OUT.err
echo "done $OUT $(date +%H:%M:%S)" >> $S/run.log
