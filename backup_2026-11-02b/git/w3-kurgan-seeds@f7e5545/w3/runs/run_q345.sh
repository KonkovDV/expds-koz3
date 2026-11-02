cd /workspace/koz3/orch/bot-W3; export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
for q in Q5 Q4 Q3; do nice -n 10 /workspace/koz3/venv/bin/python code/q345.py 2000 $q > runs/gate_$q.log 2>&1; echo $q EXIT $?; done; echo ALL_DONE
