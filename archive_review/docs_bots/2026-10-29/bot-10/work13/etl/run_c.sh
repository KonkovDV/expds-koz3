#!/bin/bash
cd /workspace/koz3_shared/bot-10
export EXPDS_METRIC=$PWD/repo/materials/compute_metrics.py OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 EXPDS_MATCH_SCOPE=global
PY=/workspace/rerank/venv/bin/python; G=/workspace/rerank/gt_train_v2.geojson
ulimit -v 3000000
for v in A S Ss6; do
 case $v in Ss6) SRC=work4/curves_g_sep;; *) SRC=work/curves;; esac
 for fl in 0 1; do
  D=work13/etl/c_${v}_$fl; rm -rf $D; mkdir -p $D
  for f in $SRC/*.pkl; do ln -s $(readlink -f $f) $D/$(basename $f); done
  rm -f $D/Гнёздово.pkl
  EXPDS_ETGR_LAYER=$fl nice -n 10 $PY repo/scripts/family_curves.py --fam work13/etl/fam_$v --gt $G --clusters work/family_clusters.json --out $D > $D.log 2>&1; echo "curves $v flag=$fl rc=$? $(date +%T)"
 done
done
echo done
