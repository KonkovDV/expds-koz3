export OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1; ulimit -v 3000000
cd /workspace/koz3_shared/bot-10/work6
nice -n 10 /workspace/rerank/venv/bin/python taus.py A /workspace/koz3_shared/bot-10/work/curves > taus_A.log 2>&1
nice -n 10 /workspace/rerank/venv/bin/python taus.py kur /workspace/koz3_shared/bot-7/work2/curves_kur > taus_kur.log 2>&1
echo done >> taus_A.log
