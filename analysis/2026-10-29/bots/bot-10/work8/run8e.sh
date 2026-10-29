#!/bin/bash
cd /workspace/koz3_shared/bot-10
export EXPDS_METRIC=$PWD/repo/materials/compute_metrics.py OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
PY=/workspace/rerank/venv/bin/python; G=/workspace/rerank/gt_train_v2.geojson
ulimit -v 3000000
echo "start $(date +%T)"
curves() { # $1 вариант, $2 база кривых, $3 каталог
  D=$3; rm -rf $D; mkdir -p $D
  for f in $2/*.pkl; do ln -s $(readlink -f $f) $D/$(basename $f); done
  $PY - $1 $D <<'PYE'
import json,sys
from pathlib import Path
v,D=sys.argv[1],sys.argv[2]; aff=set(json.load(open(f'work8/fam_{v}/_affected.json')))
cl=json.load(open('work/family_clusters.json'))['clusters']
for f in aff:
    name=f
    for m in cl.values():
        if f in m: name='+'.join(sorted(m))
    p=Path(D)/f'{name}.pkl'
    if p.is_symlink() or p.exists(): p.unlink()
PYE
  nice -n 10 $PY repo/scripts/family_curves.py --fam work8/fam_$1 --gt $G --clusters work/family_clusters.json --out $D > $D.log 2>&1
  echo "curves $1 rc=$? n=$(ls $D|wc -l) $(date +%T)"
}
for v in gorc_noc gorc_full gorc_cs; do nice -n 10 $PY work8/mk8.py $v; done
export EXPDS_MATCH_SCOPE=global
curves gorc_noc work/curves work8/curves_gorc_noc
nice -n 10 $PY work4/paired.py --boot 2000 --base A=work/curves --b noc=work8/curves_gorc_noc --out work8/paired_gorc_noc.json > work8/paired_gorc_noc.log 2>&1
echo "paired noc $(date +%T)"
export EXPDS_MATCH_SCOPE=site
curves gorc_noc work4/curves_site work8/curves_s_gorc_noc
nice -n 10 $PY work4/paired.py --boot 200 --base A=work4/curves_site --b noc=work8/curves_s_gorc_noc --out work8/paired_gorc_noc_site.json > work8/paired_gorc_noc_site.log 2>&1
echo "paired noc site $(date +%T)"
export EXPDS_MATCH_SCOPE=global
curves gorc_full work/curves work8/curves_gorc_full
curves gorc_cs work/curves work8/curves_gorc_cs
nice -n 10 $PY work4/paired.py --boot 200 --base A=work/curves --b full=work8/curves_gorc_full --b cs=work8/curves_gorc_cs --out work8/paired_gorc_ref.json > work8/paired_gorc_ref.log 2>&1
echo "done $(date +%T)"
