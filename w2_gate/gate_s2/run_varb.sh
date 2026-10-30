#!/bin/bash
set -euo pipefail
K=/workspace/koz3; G=$K/w2/gate_s2; PY=$K/venv/bin/python; N="nice -n 10"
R=$K/wt/w5/analysis/2026-10-30/w5_rerank
OOF=$K/w2/oof/merged/oof_cnx_c3-sc_s012avg.geojson
run() { # tag extra
  tag=$1; shift; D=$G/$tag
  $N $PY $K/w2/tools/stack_replace_li.py $K/w5data/stack_arch.geojson $OOF $D/stack_arch_$tag.geojson cnx012 "$@" > $G/logs/build_$tag.log 2>&1
  cd $R
  W5_DATA=$D W5_STACK=stack_arch_$tag.geojson $N $PY prep.py > $G/logs/prep_$tag.log 2>&1
  W5_DATA=$D $N $PY gate_v6c.py --vs $K/w5data/work/base_counts.pkl --save counts_$tag.pkl --out w6c_gate_$tag.json > $G/logs/gate_$tag.log 2>&1
  W5_DATA=$D PYTHONPATH=$R $N $PY $K/w2/tools/gate_detail.py $K/w5data/work/base_counts.pkl $D/work/counts_$tag.pkl $G/detail_$tag.json > $G/logs/detail_$tag.log 2>&1
  rm -f $D/stack_arch_$tag.geojson
  echo "done $tag $(date '+%H:%M')"
}
run varb
run varb_kur --classes kurgany_povrezhdennye,kurgany_tselye
