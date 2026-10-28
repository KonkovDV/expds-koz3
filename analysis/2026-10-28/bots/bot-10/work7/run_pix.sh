#!/bin/bash
cd /workspace/koz3_shared/bot-10
export EXPDS_METRIC=$PWD/repo/materials/compute_metrics.py OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 EXPDS_MATCH_SCOPE=global
PY=/workspace/rerank/venv/bin/python; G=/workspace/rerank/gt_train_v2.geojson
ulimit -v 3000000
echo "start $(date +%T)"
nice -n 10 $PY work7/mkfam_pix.py > work7/mkfam_pix.log 2>&1
for v in Sf S SfR3; do rm -rf work7/fam_${v}s6; nice -n 10 $PY work4/stack6_fam.py --src work7/fam_$v --dst work7/fam_${v}s6 --mode separate --fams Гнёздово,Семибратнее,Судьбищи > work7/s6_$v.log 2>&1; done
echo "s6 $(date +%T)"
for v in Sf S SfR3 Sfs6 Ss6 SfR3s6; do
  case $v in *s6) SRC=work4/curves_g_sep;; *) SRC=work/curves;; esac
  D=work7/curves_g_$v; rm -rf $D; mkdir -p $D
  for f in $SRC/*.pkl; do ln -s $(readlink -f $f) $D/$(basename $f); done
  rm -f $D/Гнёздово.pkl $D/Семибратнее.pkl $D/Судьбищи.pkl
  nice -n 10 $PY repo/scripts/family_curves.py --fam work7/fam_$v --gt $G --clusters work/family_clusters.json --out $D > $D.log 2>&1
done
echo "curves $(date +%T)"
nice -n 10 $PY work4/paired.py --boot 200 --base A=work/curves --b Sf=work7/curves_g_Sf --b S=work7/curves_g_S --b SfR3=work7/curves_g_SfR3 --out work7/paired_g_pix.json > work7/paired_g_pix.log 2>&1
echo "p1 $(date +%T)"
nice -n 10 $PY work4/paired.py --boot 200 --base PKG=work7/curves_g_PKG --b S=work7/curves_g_S --b SfR3=work7/curves_g_SfR3 --out work7/paired_g_vsPKG.json > work7/paired_g_vsPKG.log 2>&1
echo "p2 $(date +%T)"
nice -n 10 $PY work4/paired.py --boot 200 --base S6=work4/curves_g_sep --b Sfs6=work7/curves_g_Sfs6 --b Ss6=work7/curves_g_Ss6 --b SfR3s6=work7/curves_g_SfR3s6 --out work7/paired_g_pix_s6.json > work7/paired_g_pix_s6.log 2>&1
echo "done $(date +%T)"
