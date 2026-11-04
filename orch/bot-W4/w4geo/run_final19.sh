#!/bin/bash
cd /workspace/koz3/w4clean
E=/workspace/koz3/w4geo/eval; PY=/workspace/koz3/w4venv/bin/python
export EXPDS_METRIC=/workspace/koz3/ds/e2e/score/compute_metrics.py
R="$(ls /workspace/koz3/w4geo/train | grep -E '^2(0[0-9]|1[0-7])_' | tr '\n' ' ')"
C="nice -n 19 $PY scripts/pix_clean_oof.py compose --root /workspace/koz3/w4geo/train --config configs/v3_on.toml --cache /workspace/koz3/w4geo/oof --regions $R"
P="--set geophys.pixgbm.enabled=true --set geophys.pixgbm.impl=\"clean\" --set geophys.pixgbm.maps=[\"etgr\"] --tag s0"
$C $P --orget-tag s0 --out $E/C19etonly_orget_s0.geojson || echo FAIL C19
G="nice -n 19 $PY scripts/pix_modality_gate.py --gt /workspace/koz3/ds/gt/gt_train_v2.geojson --a $E/A18.geojson --c $E/C19etonly_orget_s0.geojson --etgr 212 213 214 215 216 217"
$G --out $E/FINAL19_gate_sweep.json
$G --fix etgr:object_poly=0 --out $E/FINAL19_gate_N0.json
sha256sum $E/C19etonly_orget_s0.geojson
echo DONE
