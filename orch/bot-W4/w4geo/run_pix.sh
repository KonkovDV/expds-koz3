#!/bin/bash
# pix_clean кандидаты по фолдам: run_pix.sh <seed> [фолды...]
s=$1; shift
cd /workspace/koz3/w4clean
T=/workspace/koz3/w4geo/train; M=/workspace/koz3/w4geo/models
declare -A F=( [F1]="^200_" [F2]="^20[1-7]_" [F3]="^(208|209|210|215)_" [F4]="^(211|212|216)_" [F5]="^(213|214|217)_" )
for k in "${@:-F1 F2 F3 F4 F5}"; do
  regs=$(ls $T | grep -E "${F[$k]}")
  OMP_NUM_THREADS=3 nice -n 10 /workspace/koz3/w4venv/bin/python scripts/pix_clean_oof.py pix --root $T --config configs/v3_on.toml \
    --set geophys.pixgbm.enabled=true --set 'geophys.pixgbm.impl="clean"' --set geophys.pixgbm.replace=true \
    --set 'geophys.pixgbm.union_name_re=""' --set geophys.pixgbm.ground_metres=false --set geophys.pixgbm.procs=3 \
    --model-dir $M/${k}_s$s --cache /workspace/koz3/w4geo/oof --tag s$s --regions $regs || echo "FAIL $k s$s"
done
echo PIXDONE s$s
