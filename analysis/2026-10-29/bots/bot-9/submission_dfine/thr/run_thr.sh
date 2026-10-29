#!/bin/bash
# Пересборка кривых R (как run5.sh) только для sweep порогов; временные каталоги удаляются.
M=/workspace/koz3_shared/bot-9/kaggle/dfine_spor/full_rest_out/metric; B10=/workspace/koz3_shared/bot-10; T=/workspace/koz3_shared/bot-9/submission_dfine/thr
export EXPDS_METRIC=$B10/repo/materials/compute_metrics.py OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 EXPDS_MATCH_SCOPE=global
PY=/workspace/rerank/venv/bin/python; CL=$B10/work/family_clusters.json; DROP=sp_arch,sp_focal,sp_focal2,sp_or
ulimit -v 3000000
echo "df $(df -h / | tail -1 | awk '{print $4}') $(date +%T) start"
cd $M && nice -n 10 $PY prep5.py gt > $T/gt.log 2>&1; echo "gt rc=$?"
cd $M && nice -n 10 $PY prep5.py fam C > $T/fam_C.log 2>&1; echo "fam C rc=$? df $(df -h / | tail -1 | awk '{print $4}')"
cd $B10 && nice -n 10 $PY repo/scripts/family_curves.py --clusters $CL --fam $M/fam_C --gt $M/gt_t.geojson --drop-groups $DROP --out $T/curves_R_t > $T/curves_R_t.log 2>&1; echo "c R_t rc=$? $(date +%T)"
find $M/fam_C -type f -name '*.jsonl' -delete; find $M/fam_C -type l -delete; rmdir $M/fam_C
mkdir -p $T/c5_R; A=$B10/work/curves
for p in $A/*.pkl; do u=$(basename $p); if [ -f $T/curves_R_t/$u ]; then ln -sf $T/curves_R_t/$u $T/c5_R/$u; else ln -sf $p $T/c5_R/$u; fi; done
echo "c5_R units $(ls $T/c5_R | wc -l), touched $(ls $T/curves_R_t/*.pkl | wc -l)"
nice -n 10 $PY $T/sweep_R.py $T/c5_R $T/thr_sweep.json > $T/sweep.log 2>&1; echo "sweep rc=$? $(date +%T)"
rm -f $M/gt_t.geojson $M/gt_t_sp.geojson
echo "df $(df -h / | tail -1 | awk '{print $4}') $(date +%T) done"
