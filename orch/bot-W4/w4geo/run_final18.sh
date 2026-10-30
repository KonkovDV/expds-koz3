#!/bin/bash
cd /workspace/koz3/w4clean
E=/workspace/koz3/w4geo/eval; PY=/workspace/koz3/w4venv/bin/python
export EXPDS_METRIC=/workspace/koz3/ds/e2e/score/compute_metrics.py
R="$(ls /workspace/koz3/w4geo/train | grep -E '^2(0[0-9]|1[0-7])_' | tr '\n' ' ')"
C="nice -n 19 $PY scripts/pix_clean_oof.py compose --root /workspace/koz3/w4geo/train --config configs/v3_on.toml --cache /workspace/koz3/w4geo/oof --regions $R"
P="--set geophys.pixgbm.enabled=true --set geophys.pixgbm.impl=\"clean\" --set geophys.pixgbm.maps=[\"etgr\",\"terfl\"] --tag s0"
$C --out $E/A18.geojson || echo FAIL A18
$C $P --orget-tag s0 --out $E/C18et_orget_s0.geojson || echo FAIL C18
nice -n 19 $PY scripts/pix_modality_gate.py --gt /workspace/koz3/ds/gt/gt_train_v2.geojson --a $E/A18.geojson \
  --c $E/C18et_orget_s0.geojson --etgr 212 213 214 215 216 217 --out $E/FINAL18_modality_gate.json
sha256sum $E/A18.geojson $E/C18et_orget_s0.geojson /workspace/koz3/regate5/eval/A18.geojson /workspace/koz3/regate5/eval/Cet_orget.geojson
echo DONE
