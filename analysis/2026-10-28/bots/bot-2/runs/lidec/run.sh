#!/bin/bash
set -u
B2=/workspace/koz3_shared/bot-2; B10=/workspace/koz3_shared/bot-10; W=$B2/work/lidec
PY=/workspace/rerank/venv/bin/python; G=/workspace/rerank/gt_train_v2.geojson
export EXPDS_METRIC=$B10/repo/materials/compute_metrics.py OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
ulimit -v 3000000
echo "start $(date +%T)"
for v in D1 D2 D3; do
  nice -n 10 $PY $B2/repo/scripts/li_decode_stack.py --src $B10/work/fam --dst $W/fam_$v --variant $v > $W/fam_$v.log 2>&1 || { echo "transform $v FAIL"; exit 1; }
  echo "fam $v $(date +%T) $(tail -1 $W/fam_$v.log)"
  (cd $B10 && nice -n 10 $PY repo/scripts/family_curves.py --fam $W/fam_$v --gt $G --clusters work/family_clusters.json --out $W/curves_$v > $W/curves_$v.log 2>&1) &
  (cd $B10 && EXPDS_MATCH_SCOPE=site nice -n 10 $PY repo/scripts/family_curves.py --fam $W/fam_$v --gt $G --clusters work/family_clusters.json --out $W/curves_site_$v > $W/curves_site_$v.log 2>&1) &
  wait
  echo "curves $v $(date +%T) $(ls $W/curves_$v | wc -l) $(ls $W/curves_site_$v | wc -l)"
  rm -rf $W/fam_$v/*.jsonl
done
cd $B10
nice -n 10 $PY work4/paired.py --pols drop --base A=work/curves --b D1=$W/curves_D1 --b D2=$W/curves_D2 --b D3=$W/curves_D3 --boot 1000 --out $W/paired_global.json > $W/paired_global.log 2>&1 &
nice -n 10 $PY work4/paired.py --pols drop --base A=work4/curves_site --b D1=$W/curves_site_D1 --b D2=$W/curves_site_D2 --b D3=$W/curves_site_D3 --boot 1000 --out $W/paired_site.json > $W/paired_site.log 2>&1 &
wait
echo "done $(date +%T)"
