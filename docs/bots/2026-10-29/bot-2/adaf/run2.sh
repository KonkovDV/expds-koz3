#!/bin/bash
# re-run of the evaluation stage only: bot-10/work/fam was deleted at 08:21 -> base = bot-10/work11/fam0 (same stack, read-only)
set -u
B2=/workspace/koz3_shared/bot-2; B10=/workspace/koz3_shared/bot-10; W=$B2/adaf; SRC=$B10/work11/fam0
PY2=$B2/venv/bin/python; PY=/workspace/rerank/venv/bin/python; G=/workspace/rerank/gt_train_v2.geojson
export MALLOC_ARENA_MAX=1 MALLOC_TRIM_THRESHOLD_=65536 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export EXPDS_METRIC=$B10/repo/materials/compute_metrics.py PYTHONDONTWRITEBYTECODE=1
ulimit -v 2500000
echo "start2 $(date +%T) disk $(df -BM --output=avail / | tail -1)"
cd $B10
[ -d $W/curves_A0 ] || { nice -n 10 $W/peak.sh $PY repo/scripts/family_curves.py --fam $SRC --gt $G --clusters work/family_clusters.json --out $W/curves_A0 > $W/curves_A0.log 2>&1 || { echo "curves A0 FAIL"; exit 1; }; }
echo "curves A0 $(date +%T) $(ls $W/curves_A0 | wc -l) files"
for v in AB ABE; do
  [ -d $W/curves_$v ] && continue
  case $v in AB) ms="OD_barrow_sd";; ABE) ms="OD_barrow_sd OD_enclosure_sd";; esac
  nice -n 10 $PY2 $B2/repo/scripts/adaf/adaf_splice.py --src $SRC --dets $W/dets --models $ms --out $W/fam_$v > $W/fam_$v.log 2>&1 || { echo "splice $v FAIL"; exit 1; }
  echo "fam $v $(date +%T) $(tail -1 $W/fam_$v.log) du $(du -sm $W/fam_$v | cut -f1)M disk $(df -BM --output=avail / | tail -1)"
  nice -n 10 $W/peak.sh $PY repo/scripts/family_curves.py --fam $W/fam_$v --gt $G --clusters work/family_clusters.json --out $W/curves_$v > $W/curves_$v.log 2>&1 || { echo "curves $v FAIL"; tail -5 $W/curves_$v.log; rm -rf $W/fam_$v; exit 1; }
  echo "curves $v $(date +%T) $(ls $W/curves_$v | wc -l) files"
  cp $W/fam_$v/_adaf_stats.json $W/stats_$v.json; rm -rf $W/fam_$v
done
nice -n 10 $W/peak.sh $PY $B2/repo/scripts/li_real_paired.py --base A=work/curves --b A0=$W/curves_A0 --b AB=$W/curves_AB --b ABE=$W/curves_ABE --boot 1000 --out $W/paired_vsA.json > $W/paired_vsA.log 2>&1 || echo "paired FAIL"
echo "done2 $(date +%T) disk $(df -BM --output=avail / | tail -1)"
