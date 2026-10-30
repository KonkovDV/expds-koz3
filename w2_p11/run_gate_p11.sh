#!/bin/bash
# P-W2-11 H1-ветка: стек A + исправленный s012avg АДДИТИВНО (stack_add_cnx, как исходный s012 B=0,307564), параметры гейта без изменений
set -euo pipefail
K=/workspace/koz3; G=$K/w2/p11/gate; PY=$K/venv/bin/python; N="nice -n 10"; R=$K/wt/w5/analysis/2026-10-30/w5_rerank
OOF=$K/w2/p11/oof_cnx_c3-sc_s012avg_p11.geojson; tag=s012p11; D=$G/$tag; mkdir -p $D $G/logs; ln -sfn $K/w5data/gt_train_v2.geojson $D/gt_train_v2.geojson
$N $PY $K/w2/tools/stack_add_cnx.py $K/w5data/stack_arch.geojson $OOF $D/stack_arch_$tag.geojson cnx_s012 > $G/logs/build_$tag.log 2>&1
cd $R
W5_DATA=$D W5_STACK=stack_arch_$tag.geojson $N $PY prep.py > $G/logs/prep_$tag.log 2>&1
W5_DATA=$D $N $PY gate_v6c.py --vs $K/w5data/work/base_counts.pkl --save counts_$tag.pkl --out w6c_gate_$tag.json > $G/logs/gate_$tag.log 2>&1
W5_DATA=$D PYTHONPATH=$R $N $PY $K/w2/tools/gate_detail.py $K/w5data/work/base_counts.pkl $D/work/counts_$tag.pkl $G/detail_$tag.json > $G/logs/detail_$tag.log 2>&1
W5_DATA=$D $N $PY gate_v6c.py --vs $K/w2/gate_s2/s012/work/counts_s012.pkl --out w6c_gate_${tag}_vs_s012.json > $G/logs/gate_${tag}_vs_s012.log 2>&1 || echo "vs_s012 failed"
rm -f $D/stack_arch_$tag.geojson; echo "done $tag $(date '+%H:%M')"
