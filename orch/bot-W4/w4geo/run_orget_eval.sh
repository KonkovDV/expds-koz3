#!/bin/bash
cd /workspace/koz3/w4clean
E=/workspace/koz3/w4geo/eval; PY=/workspace/koz3/w4venv/bin/python
export EXPDS_METRIC=/workspace/koz3/ds/e2e/score/compute_metrics.py
R="201_Семибратнее_Городище_1 202_Семибратнее_Городище_2 203_Семибратнее_Городище_3 204_Семибратнее_Городище_4 205_Семибратнее_Городище_5 206_Семибратнее_Городище_6 207_Семибратнее_Городище_7 208_Гнёздово_Пойма 209_Гнёздово_Терраса_1 210_Гнёздово_Терраса_2 211_Гнёздово_Терраса_3 212_Гнёздово_Усадьба 213_Гнёздово_Лесная_1 214_Гнёздово_Лесная_2 215_Гнёздово_Селище_1 216_Гнёздово_Селище_2 217_Гнёздово_Селище_3"
C="nice -n 19 $PY scripts/pix_clean_oof.py compose --root /workspace/koz3/w4geo/train --config configs/v3_on.toml --cache /workspace/koz3/w4geo/oof --regions $R"
P="--set geophys.pixgbm.enabled=true --set geophys.pixgbm.impl=\"clean\" --set geophys.pixgbm.maps=[\"etgr\",\"terfl\"] --tag s0"
$C --out $E/A17r.geojson || echo FAIL A17r
$C --orget-tag s0 --out $E/A17_orget_s0.geojson || echo FAIL a
$C $P --orget-tag s0 --out $E/C17et_orget_s0.geojson || echo FAIL c
$C $P --orget-tag s0 --org-tag s0 --set 'geophys.pixgbm.org_classes=["finds_points"]' --out $E/C17et_orgpts_orget_s0.geojson || echo FAIL d
$C $P --orget-tag s0 --org-tag s0 --out $E/C17et_orgboth_orget_s0.geojson || echo FAIL e
nice -n 19 $PY scripts/pix_clean_eval.py --gt /workspace/koz3/ds/gt/gt_train_v2.geojson --a $E/A17.geojson \
  --c $E/A17r.geojson $E/A17_orget_s0.geojson $E/C17et_orget_s0.geojson $E/C17et_orgpts_orget_s0.geojson $E/C17et_orgboth_orget_s0.geojson /workspace/koz3/org_repro/out/C_et_poly.geojson /workspace/koz3/org_repro/out/C_mgpoly_et.geojson --out $E/R17orget_vsA_s0.json
echo DONE
