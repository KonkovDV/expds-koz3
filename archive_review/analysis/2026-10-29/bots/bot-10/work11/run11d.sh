#!/bin/bash
# C1 calib: CAL1 global+site, затем справочные (rank, quantile, CALall) global
cd /workspace/koz3_shared/bot-10
export EXPDS_METRIC=$PWD/repo/materials/compute_metrics.py OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
PY=/workspace/rerank/venv/bin/python; G=/workspace/rerank/gt_train_v2.geojson
ulimit -v 3000000
run() { n=$1; m=$2; shift 2
  nice -n 10 $PY work11/xform_calib.py --src work11/fam0 --dst work11/fam_$n --mode $m --groups "$@" > work11/fam_$n.log 2>&1; echo "$n fam rc=$? $(date +%T)"
  EXPDS_MATCH_SCOPE=global nice -n 10 $PY repo/scripts/family_curves.py --fam work11/fam_$n --gt $G --clusters work/family_clusters.json --out work11/curves_${n}_g > work11/curves_${n}_g.log 2>&1; echo "$n cg rc=$? $(date +%T)"; [ "$n" != cal1 ] && rm -rf work11/fam_$n
  nice -n 10 $PY work4/paired.py --pols drop,keep --base A=work/curves --b $n=work11/curves_${n}_g --out work11/${n}_global.json > work11/${n}_global.log 2>&1; echo "$n pg rc=$? $(date +%T)"
}
site() { n=$1
  EXPDS_MATCH_SCOPE=site nice -n 10 $PY repo/scripts/family_curves.py --fam work11/fam_$n --gt $G --clusters work/family_clusters.json --out work11/curves_${n}_s > work11/curves_${n}_s.log 2>&1; echo "$n cs rc=$? $(date +%T)"; rm -rf work11/fam_$n
  nice -n 10 $PY work4/paired.py --pols drop,keep --base A=work4/curves_site --b $n=work11/curves_${n}_s --out work11/${n}_site.json > work11/${n}_site.log 2>&1; echo "$n ps rc=$? $(date +%T)"
}
echo "start $(date +%T)"
run cal1 mix li sp_or sp_focal
site cal1
# C3 frame prior FR1 (Дополнение 3), global + site — раньше справочных калибровок (приоритет оркестратора)
nice -n 10 $PY work11/xform_frame.py --src work11/fam0 --dst work11/fam_fr1 > work11/fam_fr1.log 2>&1; echo "fr1 fam rc=$? $(date +%T)"
EXPDS_MATCH_SCOPE=global nice -n 10 $PY repo/scripts/family_curves.py --fam work11/fam_fr1 --gt $G --clusters work/family_clusters.json --out work11/curves_fr1_g > work11/curves_fr1_g.log 2>&1; echo "fr1 cg rc=$? $(date +%T)"
nice -n 10 $PY work4/paired.py --pols drop,keep --base A=work/curves --b fr1=work11/curves_fr1_g --out work11/fr1_global.json > work11/fr1_global.log 2>&1; echo "fr1 pg rc=$? $(date +%T)"
site fr1
nice -n 10 $PY work4/paired.py --pols drop,keep --base A=work/curves --b cal1=work11/curves_cal1_g --out work11/cal1_global.json > work11/cal1_global.log 2>&1; echo "cal1 pg2 rc=$? $(date +%T)"
run cal1r rank li sp_or sp_focal
run cal1q quantile li sp_or sp_focal
run calall mix li li_dev li_mosaic sp_focal sp_focal2
echo done
