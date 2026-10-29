#!/bin/bash
# Кривые по семьям фолдов 4,2,1 и парные Δ (скрипты бота 10 только на чтение; всё пишется сюда).
M=/workspace/koz3_shared/bot-9/kaggle/dfine_spor/full_out/metric
B10=/workspace/koz3_shared/bot-10
export EXPDS_METRIC=$B10/repo/materials/compute_metrics.py OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 EXPDS_MATCH_SCOPE=global
PY=/workspace/rerank/venv/bin/python
ulimit -v 3000000
cd $B10
echo "start $(date +%T)"
for v in base C E; do nice -n 10 $PY repo/scripts/family_curves.py --fam $M/fam_$v --gt $M/gt_421.geojson --out $M/curves_$v > $M/curves_$v.log 2>&1; echo "c $v rc=$? $(date +%T)"; done
nice -n 10 $PY repo/scripts/family_curves.py --fam $M/fam_Conly --gt $M/gt_421_sp.geojson --out $M/curves_Conly_sp > $M/curves_Conly_sp.log 2>&1; echo "c Conly rc=$? $(date +%T)"
nice -n 10 $PY work4/paired.py --pols drop,keep --base A421=$M/curves_base --b C=$M/curves_C --b E=$M/curves_E --out $M/paired_421.json > $M/paired_421.log 2>&1; echo "p rc=$? $(date +%T)"
echo done
