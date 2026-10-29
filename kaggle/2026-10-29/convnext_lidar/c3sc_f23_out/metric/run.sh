#!/bin/bash
M=/workspace/koz3_shared/bot-9/kaggle/convnext_lidar/c3sc_f23_out/metric; B10=/workspace/koz3_shared/bot-10
export EXPDS_METRIC=$B10/repo/materials/compute_metrics.py OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 EXPDS_MATCH_SCOPE=global
PY=/workspace/rerank/venv/bin/python; CL=$B10/work/family_clusters.json
ulimit -v 3000000
fc() { cd $B10 || exit 1; nice -n 10 $PY repo/scripts/family_curves.py --clusters $CL "$@"; }
dfs() { echo "df $(df -h / | tail -1 | awk '{print $4}') $(date +%T) $1"; }
rmfam() { find $M/fam_$1 -type f -name '*.jsonl' -delete; find $M/fam_$1 -type l -delete; rmdir $M/fam_$1; dfs "rm fam_$1"; }
# только затронутые семьи (25 + члены кластера): fam_V и GT только их
cd $M; nice -n 10 $PY prep.py gt > gt.log 2>&1; echo "gt rc=$?"; dfs start
for V in base X Xx; do
  cd $M; nice -n 10 $PY prep.py fam $V > fam_$V.log 2>&1; echo "fam $V rc=$?"; dfs "fam $V"
  fc --fam $M/fam_$V --gt $M/gt_t.geojson --out $M/curves_${V}_t > $M/curves_${V}_t.log 2>&1; echo "c ${V}_t rc=$? $(date +%T)"
  [ $V != base ] && { fc --fam $M/fam_$V --gt $M/gt_li.geojson --out $M/curves_${V}_li > $M/curves_${V}_li.log 2>&1; echo "c ${V}_li rc=$? $(date +%T)"; }
  [ $V = base ] && { fc --fam $M/fam_$V --gt $M/gt_li.geojson --out $M/curves_${V}_li > $M/curves_${V}_li.log 2>&1; echo "c ${V}_li rc=$? $(date +%T)"; }
  rmfam $V
done
for V in cnx cnxx; do
  cd $M; nice -n 10 $PY prep.py fam $V > fam_$V.log 2>&1; echo "fam $V rc=$?"
  fc --fam $M/fam_$V --gt $M/gt_li.geojson --out $M/curves_${V}_li > $M/curves_${V}_li.log 2>&1; echo "c ${V}_li rc=$? $(date +%T)"
  rmfam $V
done
echo done
