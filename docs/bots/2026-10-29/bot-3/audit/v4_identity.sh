#!/bin/bash
# ulimit -v 2 ГиБ (как opt_runs: MemoryError на больших картах ловится по карте, одинаково в обоих деревьях).
# v4: байт-идентичность bot7-candidate-v3 (36af6b7) vs bot3-v4-patches на трейне 200/204/212/214. 2 потока, nice 10, сторож RSS 2,0 ГБ.
set -u
B=/workspace/koz3_shared/bot-3; O=$B/audit/out/v4
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 EXPDS_PIXGBM_THREADS=2 PYTHONPATH=$B/ehimen/pylib
PY=$B/venv/bin/python
R=(--root /workspace/data/train-expds-3 --regions 200_Судьбищи_2 204_Семибратнее_Городище_4 212_Гнёздово_Усадьба 214_Гнёздово_Лесная_2)
run() { n=$1; wt=$2; shift 2
  ( cd $B/$wt && ulimit -v 2097152 && TIMEFORMAT="WALL %R s" && time nice -n 10 $PY scripts/geo_predict.py --out $O/$n.geojson "${R[@]}" --set geophys.pixgbm.procs=1 "$@" ) > $O/$n.log 2>&1 &
  pid=$!; peak=0
  while kill -0 $pid 2>/dev/null; do
    for p in $pid $(pgrep -P $pid) $(pgrep -f "geo_predict.py --out $O/$n.geojson"); do r=$(ps -o rss= -p $p 2>/dev/null | tr -d ' '); r=${r:-0}; [ "$r" -gt "$peak" ] && peak=$r; [ "$r" -gt 2000000 ] && { echo "KILL rss=$r" >> $O/$n.log; pkill -f "geo_predict.py --out $O/$n.geojson"; }; done
    sleep 2; done
  wait $pid; echo "$n exit=$? peak_rss_kB=$peak" >> $O/runs.txt; }
run base_v3 tmp_v3base
run v4_default patch_v4
run v4_gm_cfg patch_v4 --set geophys.ground_metres=true
EXPDS_GROUND_M=1 run v4_gm_env patch_v4
echo ALLDONE >> $O/runs.txt
