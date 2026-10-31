#!/bin/bash
# Задача 11: оценка li_half по выгрузке с ПК. Сначала: запись в bot-2/EXPERIMENTS.md («Задача 11») + sha этого
# каталога в work/lihalf/eval/EXPERIMENTS_E11.sha256. Запуск:
#   nohup bash work/lihalf/eval/li_half_run.sh > work/lihalf/eval/run.log 2>&1 &
set -u
B2=/workspace/koz3_shared/bot-2; B10=/workspace/koz3_shared/bot-10; E=$B2/work/lihalf/eval; W=$B2/work/lihalf/run
PY2=$B2/venv/bin/python; PY=/workspace/rerank/venv/bin/python; G=/workspace/rerank/gt_train_v2.geojson
DUMP=${DUMP:-$B2/li_half_dump}
export EXPDS_METRIC=$B10/repo/materials/compute_metrics.py OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
mkdir -p $W
[ -f $E/EXPERIMENTS_E11.sha256 ] || { echo "нет предрегистрации (EXPERIMENTS_E11.sha256)"; exit 1; }
(cd $E && sha256sum -c EXPERIMENTS_E11.sha256) || { echo "скрипты изменены после предрегистрации"; exit 1; }
ls $DUMP/*.npz > /dev/null 2>&1 || { echo "нет выгрузки $DUMP/*.npz"; exit 1; }
nice -n 10 $PY2 $E/test_li_half_eval.py || { echo "юнит-тест отбора li_half не прошёл"; exit 1; }
echo "start $(date +%T) disk $(df -BM --output=avail / | tail -1) maps $(ls $DUMP/*.npz | wc -l)"
# H: стек A + новые курганы li_half (отбор 12 м перекрёстно по классу к принятым li/li_mosaic/li_dev, поправка 29.10), порог li_half — вложенно по семьям
nice -n 10 $PY2 $E/li_half_eval.py --dump $DUMP --out $W/fam_H > $W/fam_H.log 2>&1 || { echo "fam FAIL"; tail -5 $W/fam_H.log; exit 1; }
echo "fam H $(date +%T) $(tail -1 $W/fam_H.log) du $(du -sm $W/fam_H | cut -f1)M"
(cd $B10 && nice -n 10 $PY repo/scripts/family_curves.py --fam $W/fam_H --gt $G --clusters work/family_clusters.json --out $W/curves_H > $W/curves_H.log 2>&1) || { echo "curves FAIL"; tail -5 $W/curves_H.log; exit 1; }
cp $W/fam_H/_li_half_stats.json $W/stats_H.json
rm -rf $W/fam_H
echo "curves H $(date +%T) $(ls $W/curves_H | wc -l) files, disk $(df -BM --output=avail / | tail -1)"
# База A — work/curves бота 10 (тот же стек stack_arch.geojson, family_curves). П1–П5 + срезы new34/no_lidar/lidar59/eruntuna
cd $B10
nice -n 10 $PY $E/li_half_paired.py --base A=work/curves --b H=$W/curves_H --boot 1000 --out $W/paired_H.json > $W/paired_H.log 2>&1
cat $W/paired_H.log
echo "done $(date +%T) disk $(df -BM --output=avail / | tail -1)"
