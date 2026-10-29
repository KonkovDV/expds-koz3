#!/bin/bash
# Кривые только для 56 затронутых юнитов (D-FINE есть только там; группы sp_* базы тоже только там), с family_clusters как в bot-10/work/curves.
M=/workspace/koz3_shared/bot-9/kaggle/dfine_spor/full_rest_out/metric; B10=/workspace/koz3_shared/bot-10
export EXPDS_METRIC=$B10/repo/materials/compute_metrics.py OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 EXPDS_MATCH_SCOPE=global
PY=/workspace/rerank/venv/bin/python; CL=$B10/work/family_clusters.json; DROP=sp_arch,sp_focal,sp_focal2,sp_or
ulimit -v 3000000
fc() { cd $B10 || exit 1; nice -n 10 $PY repo/scripts/family_curves.py --clusters $CL "$@"; }
dfs() { echo "df $(df -h / | tail -1 | awk '{print $4}') $(date +%T) $1"; }
rmfam() { find $M/fam_$1 -type f -name '*.jsonl' -delete; find $M/fam_$1 -type l -delete; rmdir $M/fam_$1; dfs "rm fam_$1"; }
cd $M || exit 1; dfs start
nice -n 10 $PY prep5.py fam base > fam_base.log 2>&1; echo "fam base rc=$?"
fc --fam $M/fam_base --gt $M/gt_t.geojson --out $M/curves_base_t > $M/curves_base_t.log 2>&1; echo "c base_t rc=$? $(date +%T)"
fc --fam $M/fam_base --gt $M/gt_t_sp.geojson --out $M/curves_base_sp > $M/curves_base_sp.log 2>&1; echo "c base_sp rc=$? $(date +%T)"
fc --fam $M/fam_base --gt $M/gt_t.geojson --drop-groups $DROP --out $M/curves_R0_t > $M/curves_R0_t.log 2>&1; echo "c R0_t rc=$? $(date +%T)"
rmfam base
cd $M; nice -n 10 $PY prep5.py fam C > fam_C.log 2>&1; echo "fam C rc=$?"; dfs "fam C"
fc --fam $M/fam_C --gt $M/gt_t.geojson --out $M/curves_C_t > $M/curves_C_t.log 2>&1; echo "c C_t rc=$? $(date +%T)"
fc --fam $M/fam_C --gt $M/gt_t_sp.geojson --out $M/curves_C_sp > $M/curves_C_sp.log 2>&1; echo "c C_sp rc=$? $(date +%T)"
fc --fam $M/fam_C --gt $M/gt_t.geojson --drop-groups $DROP --out $M/curves_R_t > $M/curves_R_t.log 2>&1; echo "c R_t rc=$? $(date +%T)"
fc --fam $M/fam_C --gt $M/gt_t_sp.geojson --drop-groups $DROP --out $M/curves_R_sp > $M/curves_R_sp.log 2>&1; echo "c R_sp rc=$? $(date +%T)"
rmfam C
cd $M; nice -n 10 $PY prep5.py fam Conly > fam_Conly.log 2>&1; echo "fam Conly rc=$?"
fc --fam $M/fam_Conly --gt $M/gt_t_sp.geojson --out $M/curves_Conly_sp > $M/curves_Conly_sp.log 2>&1; echo "c Conly_sp rc=$? $(date +%T)"
rmfam Conly
cd $M; nice -n 10 $PY prep5.py fam E > fam_E.log 2>&1; echo "fam E rc=$?"; dfs "fam E"
fc --fam $M/fam_E --gt $M/gt_t.geojson --out $M/curves_E_t > $M/curves_E_t.log 2>&1; echo "c E_t rc=$? $(date +%T)"
rmfam E
echo done
