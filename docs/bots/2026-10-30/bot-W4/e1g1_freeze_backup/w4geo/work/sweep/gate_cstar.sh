#!/bin/bash
# usage: gate_cstar.sh NAME  (cstar/C_NAME.geojson -> honest filter + gate raw & _ht vs REF_v2)
set -u; N=$1; W=/workspace/koz3/w4geo/work; cd $W
PY="nice -n 10 /workspace/koz3/w4geo/venv/bin/python"; export EXPDS_METRIC=/workspace/koz3/wt/w4-pix/materials/compute_metrics.py
GT=/workspace/koz3/w5data/gt_train_v2.geojson; REF=/workspace/koz3/w4geo/ref/out/REF_v2.geojson; G=/workspace/koz3/wt/w4-pix/scripts/pix_geo_pr_gate.py
( $PY sweep/honest_filter.py $GT cstar/C_$N.geojson cstar/C_${N}_ht.geojson > cstar/ht_$N.log 2>&1 && \
  $PY $G --gt $GT --c cstar/C_${N}_ht.geojson --ref $REF --a A18.geojson --out cstar/gate_${N}_ht.json > cstar/gate_${N}_ht.log 2>&1 ) &
$PY $G --gt $GT --c cstar/C_$N.geojson --ref $REF --a A18.geojson --out cstar/gate_$N.json > cstar/gate_$N.log 2>&1 &
wait
