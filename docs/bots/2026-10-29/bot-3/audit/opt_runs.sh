#!/bin/bash
# Пункты 2–3: идентичность OFF/C2(1 модель)/SATf на трейне 201+214 и время S vs C2 на mock_val. 2 потока, nice 10.
set -u
A=/workspace/koz3_shared/bot-3/audit; O=$A/out/opt; mkdir -p $O
export OMP_NUM_THREADS=2 EXPDS_PIXGBM_THREADS=2 PYTHONPATH=/workspace/koz3_shared/bot-3/ehimen/pylib
PY=/workspace/koz3_shared/bot-3/venv/bin/python; M=/workspace/koz3_shared/bot-3/pixgbm/models
BASE=(--set geophys.pixgbm.procs=1 --set geophys.pixgbm.enabled=true --set geophys.pixgbm.replace=true --set "geophys.pixgbm.model_dir=\"$M/all\"")
R=(--root /workspace/data/train-expds-3 --regions 201_Семибратнее_Городище_1 214_Гнёздово_Лесная_2)
run() { n=$1; wt=$2; shift 2; ( cd $A/$wt && ulimit -v 2300000 && TIMEFORMAT="WALL %R s USER %U s SYS %S s" && time nice -n 10 $PY scripts/geo_predict.py --out $O/$n.geojson "$@" ) > $O/$n.log 2>&1; echo "$n exit=$?" >> $O/runs.txt; }
run S_wt wt "${R[@]}" "${BASE[@]}"
run OFF_opt wt_opt "${R[@]}" "${BASE[@]}"
run C2one_opt wt_opt "${R[@]}" "${BASE[@]}" --set "geophys.pixgbm.ensemble_dirs=[\"$M/all\"]"
run SATf_opt wt_opt "${R[@]}" "${BASE[@]}" --set "geophys.pixgbm.sat_finds={core_max_m2=60, flag=true, model_dir=\"$A/var/SAT/models/all\"}"
PIXGBM_SAT_CORE_M2=60 PIXGBM_SAT_FLAG=1 run SAT_wt wt "${R[@]}" --set geophys.pixgbm.procs=1 --set geophys.pixgbm.enabled=true --set geophys.pixgbm.replace=true --set "geophys.pixgbm.model_dir=\"$A/var/SAT/models/all\""
MV=(--root $A/mock_val)
run mock_S wt_opt "${MV[@]}" "${BASE[@]}"
run mock_C2 wt_opt "${MV[@]}" "${BASE[@]}" --set "geophys.pixgbm.ensemble_dirs=[\"$M/Судьбищи\", \"$M/Семибратнее\", \"$M/Гнёздово\"]"
echo ALLDONE >> $O/runs.txt
