#!/bin/bash
# P-W2-10 gate: seed-avg → замена li_cnx 4 участков в стеке s012 → prep → honest-10
set -euo pipefail
K=/workspace/koz3; DF=$K/w2/demfill; PY=$K/venv/bin/python; N="nice -n 10"
R=$K/wt/w5/analysis/2026-10-30/w5_rerank; S012=$K/w2/gate_s2/s012
REGS=015_Демидовка,021_Новенькое,186_Семибратнее_1,190_Муравлево
REP=$K/w2/oof/s2/koz3-w2-cnx-c3sc-s2f23/report_cnx_full.json
arm=$1; vs=$2; D=$DF/gate/$arm; mkdir -p $D $DF/gate/logs; ln -sf $K/w5data/gt_train_v2.geojson $D/gt_train_v2.geojson
[ -f $DF/out/merged_$arm.geojson ] || $N $PY $K/w2/tools/seed_merge_cnx.py $DF/maps/$arm $DF/out/merged_$arm.geojson --seeds s0,s1,s2 --report $REP > $DF/gate/logs/merge_$arm.log 2>&1
$N $PY $DF/swap_sites.py $S012/stack_arch_cnx_s012.geojson $DF/out/merged_$arm.geojson $D/stack_$arm.geojson dfill$arm $REGS > $DF/gate/logs/swap_$arm.log 2>&1
cd $R
W5_DATA=$D W5_STACK=stack_$arm.geojson $N $PY prep.py > $DF/gate/logs/prep_$arm.log 2>&1
W5_DATA=$D $N $PY gate_v6c.py --vs $vs --save counts_$arm.pkl --out w6c_gate_$arm.json > $DF/gate/logs/gate_$arm.log 2>&1
W5_DATA=$D PYTHONPATH=$R $N $PY $K/w2/tools/gate_detail.py $vs $D/work/counts_$arm.pkl $DF/gate/detail_${arm}.json > $DF/gate/logs/detail_$arm.log 2>&1
rm -f $D/stack_$arm.geojson $D/work/geom_wkb.pkl $D/work/table.pkl
echo "done $arm $(date +%H:%M)"
