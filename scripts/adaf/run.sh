#!/bin/bash
# ADAF zero-shot (pre-declared in bot-2/EXPERIMENTS.md "Задача 10"). nohup bash adaf/run.sh > adaf/run.log 2>&1 &
set -u
B2=/workspace/koz3_shared/bot-2; B10=/workspace/koz3_shared/bot-10; W=$B2/adaf
PY2=$B2/venv/bin/python; PY=/workspace/rerank/venv/bin/python; G=/workspace/rerank/gt_train_v2.geojson
export EXPDS_METRIC=$B10/repo/materials/compute_metrics.py PYTHONDONTWRITEBYTECODE=1
ulimit -v 3000000
thr() { if pgrep -f 'work/lireal/run' >/dev/null 2>&1; then echo 1; else echo 2; fi; }   # 1 thread while any bot-2 lireal run*.sh is alive (resume 07:35 changed its PID)
while ! grep -q '^done' $W/prep_lowmem.log || [ ! -f $W/weights/OD_enclosure_sd.pt ]; do sleep 30; done   # SLRM prep + weights first (one bot-2 job at a time)
echo "start $(date +%T) disk $(df -BM --output=avail / | tail -1)"
for M in OD_barrow_sd OD_enclosure_sd; do
  for f in $W/slrm/*.npz; do
    r=$(basename $f .npz); [ -f $W/dets/$M/$r.json ] && continue
    t=$(thr); OMP_NUM_THREADS=$t OPENBLAS_NUM_THREADS=$t MKL_NUM_THREADS=$t nice -n 10 $PY2 $B2/repo/scripts/adaf/infer_od.py \
      --slrm $W/slrm --weights $W/weights/$M.pt --out $W/dets --tile 1024 --stride 896 --min-size 800 --only $r \
      | sed "s/^/thr=$t /" || echo "infer FAIL $M $r"
  done
done
echo "infer done $(date +%T)"
export OMP_NUM_THREADS=$(thr) OPENBLAS_NUM_THREADS=$(thr) MKL_NUM_THREADS=$(thr)
for v in AB ABE; do
  case $v in AB) ms="OD_barrow_sd";; ABE) ms="OD_barrow_sd OD_enclosure_sd";; esac
  nice -n 10 $PY2 $B2/repo/scripts/adaf/adaf_splice.py --src $B10/work/fam --dets $W/dets --models $ms --out $W/fam_$v > $W/fam_$v.log 2>&1 || { echo "splice $v FAIL"; exit 1; }
  echo "fam $v $(date +%T) $(tail -1 $W/fam_$v.log) du $(du -sm $W/fam_$v | cut -f1)M"
  (cd $B10 && nice -n 10 $PY repo/scripts/family_curves.py --fam $W/fam_$v --gt $G --clusters work/family_clusters.json --out $W/curves_$v > $W/curves_$v.log 2>&1) || { echo "curves $v FAIL"; tail -5 $W/curves_$v.log; exit 1; }
  echo "curves $v $(date +%T) $(ls $W/curves_$v | wc -l) files"
  cp $W/fam_$v/_adaf_stats.json $W/stats_$v.json; rm -rf $W/fam_$v
done
cd $B10
nice -n 10 $PY $B2/repo/scripts/li_real_paired.py --base A=work/curves --b AB=$W/curves_AB --b ABE=$W/curves_ABE --boot 1000 --out $W/paired_vsA.json > $W/paired_vsA.log 2>&1
nice -n 10 $PY2 $B2/repo/scripts/adaf/adaf_recall.py --slrm $W/slrm --dets $W/dets --gt $G --out $W/recall.json > $W/recall.log 2>&1
echo "done $(date +%T) disk $(df -BM --output=avail / | tail -1)"
