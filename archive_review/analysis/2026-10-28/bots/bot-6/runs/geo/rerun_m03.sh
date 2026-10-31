#!/bin/bash
# SM/UM t0.3 без op_counts (запущены до правки geo_val.py) — повтор; затем сводка и проверка растров ae.
cd /workspace/koz3_shared/bot-6/work/geo
export OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 GDAL_NUM_THREADS=2 GDAL_CACHEMAX=128
PY=/workspace/rerank/venv/bin/python
for m in S U; do
  nice -n 10 $PY geo_val.py --pix $m --tpix 0.3 --markups etgr,terfl,maggr --boot 1000 --out geo_val_pix${m}M_t0.3.json 2>&1 | grep -v config.py > geo_val_pix${m}M_t0.3.log
  echo "$(date +%T) M t0.3 $m done"
done
nice -n 10 $PY compare_tpix.py 2>&1 | grep -v config.py > compare_tpix.log; echo "$(date +%T) compare done"
cd ../aefix && prlimit --as=2000000000 nice -n 10 /workspace/koz3_shared/bot-6/venv/bin/python real_check.py > real_check.log 2>&1; echo "$(date +%T) real_check done rc=$?"
echo ALLDONE
