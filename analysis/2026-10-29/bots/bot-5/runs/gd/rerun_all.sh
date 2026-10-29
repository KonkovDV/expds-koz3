#!/bin/bash
# bot-5: повтор всех замеров после пропажи bot-8/venv и e2e/out (29.10 ~13:12–13:36 MSK); python = bot-7/venv (torch 2.11 cpu, pytest 9.2)
export OMP_NUM_THREADS=2
PY=/workspace/koz3_shared/bot-7/venv/bin/python
W=/workspace/koz3_shared/bot-5/work/gd; R=/workspace/koz3_shared/bot-5/repo
cd $R && echo "== pytest $(date +%T)" && $PY $W/watch.py $W/pytest_b5.log -- taskset -c 0,1 $PY -m pytest -q tests/test_bot5_gor_dense.py tests/test_bot7_v3_flags.py; tail -n 3 $W/pytest_b5.log
cd $W/e2e && echo "== off $(date +%T)" && ./run_off.sh
for reg in 084 081 210 217; do cmp out/base/$reg.geojson out/off/$reg.geojson && echo "$reg IDENTICAL $(md5sum < out/off/$reg.geojson | cut -c1-32) $(stat -c%s out/off/$reg.geojson)B"; done
echo "== on $(date +%T)"; mkdir -p out/on
for reg in 084 081; do $PY $W/watch.py out/on/$reg.log -- taskset -c 0,1 $PY on/inference/main.py in/$reg out/on/$reg.geojson; tail -n 1 out/on/$reg.log; done
cmp out/on/084.geojson out/off/084.geojson && echo "084 ON==OFF"
cd $W && echo "== timing $(date +%T)" && $PY watch.py time_phase.log -- taskset -c 0,1 $PY time_phase.py; tail -n 1 time_phase.log
echo "== nomodel $(date +%T)" && $PY watch.py time_nomodel.log -- taskset -c 0,1 $PY time_nomodel.py; tail -n 2 time_nomodel.log
echo "RERUN_DONE $(date +%T)"
