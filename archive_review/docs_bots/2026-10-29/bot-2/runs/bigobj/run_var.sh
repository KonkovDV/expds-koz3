#!/bin/bash
export EXPDS_METRIC=/workspace/koz3_shared/bot-10/repo/materials/compute_metrics.py OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
PY=/workspace/rerank/venv/bin/python; C=/workspace/koz3_shared/bot-2/work/bigobj/code
F1='{"big_groups":{"min_area_m2":5000,"classes":["gorodishcha","fortifikatsii"]}}'
F2='{"core":{"gorodishcha":{"frac":0.3,"min_r_m":8,"min_area_m2":5000}}}'
F12='{"big_groups":{"min_area_m2":5000,"classes":["gorodishcha","fortifikatsii"]},"core":{"gorodishcha":{"frac":0.3,"min_r_m":8,"min_area_m2":5000}}}'
F1s='{"big_groups":{"min_area_m2":2000,"classes":["gorodishcha","fortifikatsii"]}}'
for v in F1 F2 F12 F1s; do
  echo "== $v $(date '+%T')"; nice -n 10 $PY var_curves.py --code $C --variant "${!v}" --out curves_$v > log_curves_$v.txt 2>&1 || echo FAIL $v
  tail -1 log_curves_$v.txt
done
echo "== done $(date '+%T')"
