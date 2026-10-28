#!/bin/bash
cd /workspace/koz3_shared/bot-10
export EXPDS_METRIC=$PWD/repo/materials/compute_metrics.py OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
PY=/workspace/rerank/venv/bin/python; G=/workspace/rerank/gt_train_v2.geojson
ulimit -v 3000000
echo "start $(date +%T)"
for v in A2 R3 PKG; do rm -rf work7/fam_${v}s6; nice -n 10 $PY work4/stack6_fam.py --src work7/fam_$v --dst work7/fam_${v}s6 --mode separate --fams Гнёздово,Семибратнее > work7/s6_$v.log 2>&1; done
echo "s6 $(date +%T)"
for sc in g s; do
  if [ $sc = g ]; then BASE=work/curves; S6=work4/curves_g_sep; export EXPDS_MATCH_SCOPE=global; else BASE=work4/curves_site; S6=work4/curves_s_sep; export EXPDS_MATCH_SCOPE=site; fi
  for v in FB RB A2 R3 PKG A2s6 R3s6 PKGs6; do
    case $v in *s6) SRC=$S6;; *) SRC=$BASE;; esac
    D=work7/curves_${sc}_$v; rm -rf $D; mkdir -p $D
    for f in $SRC/*.pkl; do ln -s $(readlink -f $f) $D/$(basename $f); done
    rm -f $D/Гнёздово.pkl $D/Семибратнее.pkl
    nice -n 10 $PY repo/scripts/family_curves.py --fam work7/fam_$v --gt $G --clusters work/family_clusters.json --out $D > $D.log 2>&1
  done
  echo "curves $sc $(date +%T)"
  nice -n 10 $PY work4/paired.py --base A=$BASE --b FB=work7/curves_${sc}_FB --b RB=work7/curves_${sc}_RB --b A2=work7/curves_${sc}_A2 --b R3=work7/curves_${sc}_R3 --b PKG=work7/curves_${sc}_PKG --out work7/paired_${sc}.json > work7/paired_${sc}.log 2>&1
  nice -n 10 $PY work4/paired.py --base S6=$S6 --b A2s6=work7/curves_${sc}_A2s6 --b R3s6=work7/curves_${sc}_R3s6 --b PKGs6=work7/curves_${sc}_PKGs6 --out work7/paired_${sc}_s6.json > work7/paired_${sc}_s6.log 2>&1
  echo "paired $sc $(date +%T)"
done
echo "done $(date +%T)"
