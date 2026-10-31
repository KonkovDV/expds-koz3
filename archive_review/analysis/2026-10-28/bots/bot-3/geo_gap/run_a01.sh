#!/bin/bash
# audit01: maggr-прогоны на участках 201–212 (200 отдельно: карта 14084×9034, >2 ГБ).
# 1 поток, nice 10, ulimit -v 3 ГБ виртуальной (RSS ≤ ~2 ГБ).
set -u
cd /workspace/koz3_shared/bot-3/repo_audit01
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
ulimit -v 3000000
OUT=/workspace/koz3_shared/bot-3/geo_gap/a01
REGS="${REGS:-201_Семибратнее_Городище_1 202_Семибратнее_Городище_2 203_Семибратнее_Городище_3 204_Семибратнее_Городище_4 205_Семибратнее_Городище_5 206_Семибратнее_Городище_6 207_Семибратнее_Городище_7 208_Гнёздово_Пойма 209_Гнёздово_Терраса_1 210_Гнёздово_Терраса_2 211_Гнёздово_Терраса_3 212_Гнёздово_Усадьба}"
COMMON="--root /workspace/data/train-expds-3 --set geophys.etgr.classic=false --set geophys.terfl.classic=false"
run() { name=$1; shift
  nice -n 10 ../venv/bin/python scripts/geo_predict.py $COMMON --out $OUT/$name.geojson "$@" --regions $REGS 2>&1 | grep -v 'рамка LAS' > $OUT/$name.log
  echo "$name exit=${PIPESTATUS[0]}" >> $OUT/done.txt; }
for c in ${CFGS:-B A1 A2 A3}; do case $c in
 B)  run B ;;
 A1) run A1 --set 'geophys.maggr.maps="all"' ;;
 A2) run A2 --set 'geophys.maggr.maps="all"' --set geophys.maggr.dedupe_maps=false ;;
 A3) run A3 --set 'geophys.maggr.maps="all"' --set geophys.maggr.dedupe_maps=false --set 'geophys.maggr.finds_points.band_mode="auto"' --set 'geophys.maggr.object_poly.band_mode="auto"' ;;
 A3m) run A3m --set 'geophys.maggr.maps="all"' --set geophys.maggr.dedupe_maps=false --set 'geophys.maggr.band_mode="auto"' ;;
esac; done
