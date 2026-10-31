#!/bin/bash
A=/workspace/koz3_shared/bot-3/audit
while pgrep -f chain1.sh >/dev/null; do sleep 30; done
$A/run_a.sh calib_train $A/calib_train.py
$A/run_a.sh calib_predict $A/calib_predict.py
$A/run_a.sh crs_identity $A/crs_identity.py
# сквозная проверка: прогноз 3 участков с EXPDS_GROUND_M=1 (модели S) → сравнение с pix_lofo
EXPDS_GROUND_M=1 $A/run_a.sh crs_predict scripts/pixgbm.py predict --root /workspace/data/train-expds-3 --models /workspace/koz3_shared/bot-3/pixgbm/models --out $A/out/crs_M_pred.geojson --regions 200_Судьбищи_2 204_Семибратнее_Городище_4 212_Гнёздово_Усадьба
A2=1; $A/run_a.sh calib_shift_px $A/calib_shift_px.py
echo DONE chain2 $(date) >> $A/out/pipe.log
