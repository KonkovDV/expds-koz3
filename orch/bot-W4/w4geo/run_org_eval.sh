#!/bin/bash
cd /workspace/koz3/w4clean
E=/workspace/koz3/w4geo/eval; PY=/workspace/koz3/w4venv/bin/python
export EXPDS_METRIC=/workspace/koz3/ds/e2e/score/compute_metrics.py
for v in both:'["finds_points","object_poly"]' poly:'["object_poly"]' pts:'["finds_points"]'; do
  n=${v%%:*}; c=${v#*:}
  OMP_NUM_THREADS=3 nice -n 19 $PY scripts/pix_clean_oof.py compose --root /workspace/koz3/w4geo/train --config configs/v3_on.toml \
    --set geophys.pixgbm.enabled=true --set 'geophys.pixgbm.impl="clean"' --set 'geophys.pixgbm.maps=["etgr","terfl"]' \
    --set "geophys.pixgbm.org_classes=$c" --cache /workspace/koz3/w4geo/oof --tag s0 --org-tag s0 --regions 201_Семибратнее_Городище_1 202_Семибратнее_Городище_2 203_Семибратнее_Городище_3 204_Семибратнее_Городище_4 205_Семибратнее_Городище_5 206_Семибратнее_Городище_6 207_Семибратнее_Городище_7 208_Гнёздово_Пойма 209_Гнёздово_Терраса_1 210_Гнёздово_Терраса_2 211_Гнёздово_Терраса_3 212_Гнёздово_Усадьба 213_Гнёздово_Лесная_1 214_Гнёздово_Лесная_2 215_Гнёздово_Селище_1 216_Гнёздово_Селище_2 217_Гнёздово_Селище_3  --out $E/C17et_org${n}_s0.geojson || echo FAIL $n
done
nice -n 19 $PY scripts/pix_clean_eval.py --gt /workspace/koz3/ds/gt/gt_train_v2.geojson --a $E/C17et_s0.geojson \
  --c $E/C17et_orgboth_s0.geojson $E/C17et_orgpoly_s0.geojson $E/C17et_orgpts_s0.geojson --out $E/R17org_vsCet_s0.json
nice -n 19 $PY scripts/pix_clean_eval.py --gt /workspace/koz3/ds/gt/gt_train_v2.geojson --a $E/A17.geojson \
  --c $E/C17et_orgboth_s0.geojson --out $E/R17org_vsA_s0.json
echo ORGEVALDONE
