#!/bin/bash
# Задача 9: реальные карты li. Запуск: nohup bash work/lireal/run.sh > work/lireal/run.log 2>&1 &
set -u
B2=/workspace/koz3_shared/bot-2; B10=/workspace/koz3_shared/bot-10; W=$B2/work/lireal
PY2=$B2/venv/bin/python; PY=/workspace/rerank/venv/bin/python; G=/workspace/rerank/gt_train_v2.geojson
PR=$B2/li_dump/li_hard2_probs
export EXPDS_METRIC=$B10/repo/materials/compute_metrics.py OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
ulimit -v 3000000
echo "start $(date +%T) disk $(df -BM --output=avail / | tail -1)"
for v in L P D1 D2 D3 CH; do
  case $v in L) m=legacy;; P) m=peaks;; D1) m=gfpoint;; D2) m=kurdual;; D3) m=hybrid;; CH) m=ch;; esac
  extra=""; [ $v = L ] && extra="--compact $W/legacy_compact.npz"
  nice -n 10 $PY2 $B2/repo/scripts/li_real_eval.py --mode $m --probs $PR --src $B10/work/fam --out $W/fam_$v $extra > $W/fam_$v.log 2>&1 || { echo "decode $v FAIL"; tail -5 $W/fam_$v.log; exit 1; }
  echo "fam $v $(date +%T) $(tail -1 $W/fam_$v.log) du $(du -sm $W/fam_$v | cut -f1)M"
  (cd $B10 && nice -n 10 $PY repo/scripts/family_curves.py --fam $W/fam_$v --gt $G --clusters work/family_clusters.json --out $W/curves_$v > $W/curves_$v.log 2>&1) || { echo "curves $v FAIL"; tail -5 $W/curves_$v.log; exit 1; }
  echo "curves $v $(date +%T) $(ls $W/curves_$v | wc -l) files, disk $(df -BM --output=avail / | tail -1)"
  cp $W/fam_$v/_li_real_stats.json $W/stats_$v.json
  rm -rf $W/fam_$v
done
cd $B10
nice -n 10 $PY $B2/repo/scripts/li_real_paired.py --base L=$W/curves_L --b P=$W/curves_P --b D1=$W/curves_D1 --b D2=$W/curves_D2 --b D3=$W/curves_D3 --b CH=$W/curves_CH --boot 1000 --out $W/paired_vsL.json > $W/paired_vsL.log 2>&1
nice -n 10 $PY $B2/repo/scripts/li_real_paired.py --base A=work/curves --b L=$W/curves_L --b P=$W/curves_P --b D1=$W/curves_D1 --b D2=$W/curves_D2 --b D3=$W/curves_D3 --b CH=$W/curves_CH --boot 1000 --out $W/paired_vsA.json > $W/paired_vsA.log 2>&1
echo "done $(date +%T) disk $(df -BM --output=avail / | tail -1)"
