#!/bin/bash
# Воспроизведение всех оценок bot-3 (1 поток, nice 10). Запуск из bot-3/geo_gap.
set -e; cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY="nice -n 10 ../venv/bin/python"
$PY check_base.py   > check_base.out        # база 0,314615 / 0,296451 / 0,144274
./run_a01.sh                                # аудит 01: a01/{B,A1,A2,A3,A3m}.geojson
$PY eval_a01.py     > a01/eval_all.out
$PY gap_poly.py                             # → gap_poly.json (причины FN object_poly)
$PY eval_geom.py    > eval_geom.out         # G1–G3
$PY eval_anchor.py  > eval_anchor.out       # H1–H3, A2, A2+H1
CFGS="RB R3 R4 R2 R1" ./run_r.sh            # растровые варианты bot3-geophys → r/*.geojson
$PY eval_r.py R3 R4 R4H1 R2 R1 > r/eval_r.out
$PY eval_pkg.py     > eval_pkg.out          # A2 finds + R3 object_poly
