cd /workspace/koz3/orch/bot-W3; export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
nice -n 10 /workspace/koz3/venv/bin/python code/kdual_dmg.py 2000; echo KD_EXIT $?
