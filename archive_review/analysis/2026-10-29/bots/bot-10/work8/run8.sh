#!/bin/bash
cd /workspace/koz3_shared/bot-10
export EXPDS_METRIC=$PWD/repo/materials/compute_metrics.py OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 EXPDS_MATCH_SCOPE=global
PY=/workspace/rerank/venv/bin/python; G=/workspace/rerank/gt_train_v2.geojson
ulimit -v 3000000
curves() { # $1 variant $2 base curves dir
  D=work8/curves_$1; rm -rf $D; mkdir -p $D
  for f in $2/*.pkl; do ln -s $(readlink -f $f) $D/$(basename $f); done
  $PY - $1 <<'PYE'
import json,sys
from pathlib import Path
v=sys.argv[1]; aff=set(json.load(open(f'work8/fam_{v}/_affected.json')))
cl=json.load(open('work/family_clusters.json'))['clusters']
for f in aff:
    name=f
    for m in cl.values():
        if f in m: name='+'.join(sorted(m))
    p=Path(f'work8/curves_{v}/{name}.pkl')
    if p.is_symlink() or p.exists(): p.unlink()
PYE
  nice -n 10 $PY repo/scripts/family_curves.py --fam work8/fam_$1 --gt $G --clusters work/family_clusters.json --out $D > $D.log 2>&1
  echo "curves $1 $(ls $D | wc -l) $(date +%T)"
}
echo "start $(date +%T)"
for v in hull hulls6 gor_noc gor_full gor_cs; do nice -n 10 $PY work8/mk8.py $v >> work8/mk8.log 2>&1; done
curves hull work/curves; curves hulls6 work4/curves_g_sep
nice -n 10 $PY work4/paired.py --boot 200 --base A=work/curves --b hull=work8/curves_hull --out work8/paired_hull.json > work8/paired_hull.log 2>&1
nice -n 10 $PY work4/paired.py --boot 200 --base S6=work4/curves_g_sep --b hulls6=work8/curves_hulls6 --out work8/paired_hull_s6.json > work8/paired_hull_s6.log 2>&1
echo "hull done $(date +%T)"
curves gor_noc work/curves; curves gor_full work/curves; curves gor_cs work/curves
nice -n 10 $PY work4/paired.py --boot 200 --base A=work/curves --b noc=work8/curves_gor_noc --b full=work8/curves_gor_full --b cs=work8/curves_gor_cs --out work8/paired_gor.json > work8/paired_gor.log 2>&1
echo "done $(date +%T)"
cd /workspace/koz3_shared/bot-10/work8
export OMP_NUM_THREADS=1
nice -n 10 $PY twins78.py global twins78_global.pkl > twins78_global.log 2>&1
nice -n 10 $PY ../work6/stats_twins.py twins78_global.pkl twins78_stats.json > twins78_stats.log 2>&1
echo "twins done $(date +%T)"
cd /workspace/koz3_shared/bot-6/work/geo && OMP_NUM_THREADS=2 nice -n 10 $PY geo_val.py --out /workspace/koz3_shared/bot-10/work8/geo_val_b6.json > /workspace/koz3_shared/bot-10/work8/geo_val_b6.log 2>&1
echo "geo_val done $(date +%T)"
