#!/bin/bash
cd /workspace/koz3_shared/bot-10
export EXPDS_METRIC=$PWD/repo/materials/compute_metrics.py OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 EXPDS_MATCH_SCOPE=global
PY=/workspace/rerank/venv/bin/python; G=/workspace/rerank/gt_train_v2.geojson
ulimit -v 3000000
echo "start $(date +%T)"
while pgrep -f "run13a" >/dev/null; do sleep 15; done; echo "run13a gone $(date +%T)"
nice -n 10 $PY work13/mkfam_b.py > work13/mkfam_b.log 2>&1; echo "mkfam rc=$? $(date +%T)"
for v in S C2 SATf; do rm -rf work13/fam_${v}s6; nice -n 10 $PY work4/stack6_fam.py --src work13/fam_$v --dst work13/fam_${v}s6 --mode separate --fams Гнёздово,Семибратнее,Судьбищи > work13/s6_$v.log 2>&1; echo "s6 $v rc=$? $(date +%T)"; done
for sc in g s; do
 for v in S C2 SATf Ss6 C2s6 SATfs6; do
  case $v in *s6) SRC=$([ $sc = g ] && echo work4/curves_g_sep || echo work4/curves_s_sep);; *) SRC=$([ $sc = g ] && echo work/curves || echo work4/curves_site);; esac
  D=work13/curves_${sc}_$v; rm -rf $D; mkdir -p $D
  for f in $SRC/*.pkl; do ln -s $(readlink -f $f) $D/$(basename $f); done
  rm -f $D/Гнёздово.pkl $D/Семибратнее.pkl $D/Судьбищи.pkl
  SCOPE=$([ $sc = g ] && echo global || echo site)
  EXPDS_MATCH_SCOPE=$SCOPE nice -n 10 $PY repo/scripts/family_curves.py --fam work13/fam_$v --gt $G --clusters work/family_clusters.json --out $D > $D.log 2>&1; echo "curves $sc $v rc=$? $(date +%T)"
 done
done
rm -rf work13/fam_S work13/fam_C2 work13/fam_SATf work13/fam_Ss6 work13/fam_C2s6 work13/fam_SATfs6; echo "fam cleaned"
for sc in g s; do
 BA=$([ $sc = g ] && echo work/curves || echo work4/curves_site)
 nice -n 10 $PY work4/paired.py --base S=work13/curves_${sc}_S --b C2=work13/curves_${sc}_C2 --b SATf=work13/curves_${sc}_SATf --out work13/paired_${sc}_vsS.json > work13/paired_${sc}_vsS.log 2>&1; echo "p vsS $sc rc=$? $(date +%T)"
 nice -n 10 $PY work4/paired.py --base Ss6=work13/curves_${sc}_Ss6 --b C2s6=work13/curves_${sc}_C2s6 --b SATfs6=work13/curves_${sc}_SATfs6 --out work13/paired_${sc}_vsSs6.json > work13/paired_${sc}_vsSs6.log 2>&1; echo "p vsSs6 $sc rc=$? $(date +%T)"
 nice -n 10 $PY work4/paired.py --base A=$BA --b S=work13/curves_${sc}_S --b C2=work13/curves_${sc}_C2 --b SATf=work13/curves_${sc}_SATf --b Ss6=work13/curves_${sc}_Ss6 --b C2s6=work13/curves_${sc}_C2s6 --b SATfs6=work13/curves_${sc}_SATfs6 --out work13/paired_${sc}_vsA.json > work13/paired_${sc}_vsA.log 2>&1; echo "p vsA $sc rc=$? $(date +%T)"
done
nice -n 10 $PY work10/rw/rw_paired.py --base S=work13/curves_g_S --b C2=work13/curves_g_C2 --b SATf=work13/curves_g_SATf --pols keep --out work13/rw_vsS.json > work13/rw_vsS.log 2>&1; echo "rw rc=$? $(date +%T)"
nice -n 10 $PY work10/rw/rw_paired.py --base Ss6=work13/curves_g_Ss6 --b C2s6=work13/curves_g_C2s6 --b SATfs6=work13/curves_g_SATfs6 --pols keep --out work13/rw_vsSs6.json > work13/rw_vsSs6.log 2>&1; echo "rw6 rc=$? $(date +%T)"
echo done
