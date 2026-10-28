#!/bin/bash
# Пересчёт pixgbm по одному уровню t_pix (прежние geo_val_pix{S,U}.json смешивали три вложенных уровня — недействительны).
cd /workspace/koz3_shared/bot-6/work/geo
export OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
PY=/workspace/rerank/venv/bin/python
nice -n 10 $PY geo_val.py --boot 1000 --out geo_val.json 2>&1 | grep -v config.py > geo_val.log
for t in 0.3 0.5 0.7; do for m in S U; do
  nice -n 10 $PY geo_val.py --pix $m --tpix $t --boot 1000 --out geo_val_pix${m}_t$t.json 2>&1 | grep -v config.py > geo_val_pix${m}_t$t.log
  nice -n 10 $PY geo_val.py --pix $m --tpix $t --markups etgr,terfl,maggr --boot 1000 --out geo_val_pix${m}M_t$t.json 2>&1 | grep -v config.py > geo_val_pix${m}M_t$t.log
  echo "$(date +%T) t$t $m done"
done; done
echo ALLDONE
