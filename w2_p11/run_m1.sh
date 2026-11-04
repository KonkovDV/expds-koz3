#!/bin/bash
# P-W2-11 M1/M3: CPU-инференс c3-sc весами (infer_cpu.py из demfill = predict_region ядра) по Kaggle-рельефу cnx_relief
# usage: run_m1.sh <seedtag> <fold> <weights.pt> <site1> [site2 ...]   → maps/<seedtag>/maps_cnx_<site>.npz
K=/workspace/koz3; P=$K/w2/p11; R=$K/w2/relief/cnx_relief
sd=$1; f=$2; w=$3; shift 3; mkdir -p $P/maps/$sd
for s in "$@"; do o=$P/maps/$sd/maps_cnx_$s.npz; [ -f $o ] && continue
  echo "== $(date +%T) $sd f$f $s"
  OMP_NUM_THREADS=3 nice -n 10 $K/venv/bin/python $K/w2/demfill/infer_cpu.py --relief $R/$s.npy --meta $R/$s.json --weights $w --fold $f --out $o --threads 3 2>&1 | grep -E "^wrote|Error|error" | tail -2
done; echo "ALLDONE $sd $(date +%T)"
