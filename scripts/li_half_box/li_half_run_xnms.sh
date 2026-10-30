#!/bin/bash
# Задача 11, ВТОРИЧНЫЙ (диагностический) вариант xnms12 — поправка 29.10 09:40 MSK. Основной результат — H (li_half_run.sh).
# HX = H + NMS 12 м внутри новых кругов li_half перекрёстно по классу (после отбора по якорям). Запускать ПОСЛЕ li_half_run.sh:
#   nohup bash work/lihalf/eval/li_half_run_xnms.sh > work/lihalf/eval/run_xnms.log 2>&1 &
set -u
B2=/workspace/koz3_shared/bot-2; B10=/workspace/koz3_shared/bot-10; E=$B2/work/lihalf/eval; W=$B2/work/lihalf/run
PY2=$B2/venv/bin/python; PY=/workspace/rerank/venv/bin/python; G=/workspace/rerank/gt_train_v2.geojson
DUMP=${DUMP:-$B2/li_half_dump}
export EXPDS_METRIC=$B10/repo/materials/compute_metrics.py OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
mkdir -p $W
(cd $E && sha256sum -c EXPERIMENTS_E11.sha256) || { echo "скрипты изменены после предрегистрации"; exit 1; }
ls $DUMP/*.npz > /dev/null 2>&1 || { echo "нет выгрузки $DUMP/*.npz"; exit 1; }
nice -n 10 $PY2 $E/test_li_half_eval.py || { echo "юнит-тест не прошёл"; exit 1; }
echo "start HX $(date +%T) disk $(df -BM --output=avail / | tail -1)"
nice -n 10 $PY2 $E/li_half_eval.py --dump $DUMP --out $W/fam_HX --self-xnms-m 12 > $W/fam_HX.log 2>&1 || { echo "fam FAIL"; tail -5 $W/fam_HX.log; exit 1; }
echo "fam HX $(date +%T) $(tail -1 $W/fam_HX.log) du $(du -sm $W/fam_HX | cut -f1)M"
(cd $B10 && nice -n 10 $PY repo/scripts/family_curves.py --fam $W/fam_HX --gt $G --clusters work/family_clusters.json --out $W/curves_HX > $W/curves_HX.log 2>&1) || { echo "curves FAIL"; tail -5 $W/curves_HX.log; exit 1; }
cp $W/fam_HX/_li_half_stats.json $W/stats_HX.json
rm -rf $W/fam_HX
cd $B10
nice -n 10 $PY $E/li_half_paired.py --base A=work/curves --b HX=$W/curves_HX --boot 1000 --out $W/paired_HX.json > $W/paired_HX.log 2>&1
cat $W/paired_HX.log
if [ -d $W/curves_H ]; then
  nice -n 10 $PY $E/li_half_paired.py --base H=$W/curves_H --b HX=$W/curves_HX --boot 1000 --out $W/paired_HvsHX.json > $W/paired_HvsHX.log 2>&1
  cat $W/paired_HvsHX.log
fi
echo "done HX $(date +%T)"
