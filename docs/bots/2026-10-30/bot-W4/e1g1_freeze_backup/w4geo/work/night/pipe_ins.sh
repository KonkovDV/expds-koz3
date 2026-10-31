#!/bin/bash
# in-sample (full-train ALL_s0 for every region, like REF) -> dump -> decode small raw -> compose -> chain -> E1 merge -> M3 rules -> gate
set -u; N=ins; W=/workspace/koz3/w4geo/work; cd $W; PY="nice -n 10 $W/../venv/bin/python"
log(){ echo "$(date '+%H:%M:%S') [$N] $*" | tee -a night/pipe.log; }
while [ ! -f models_n40r8full/ALL_s0/finds.txt ]; do sleep 30; done; log "full fit present"
$PY sweep/finds_sweep.py dump --models models_n40r8ins --probs probs_ins --threads 3 > night/dump_ins.log 2>&1; log dump
$PY sweep/finds_sweep.py decode --probs probs_ins --prefix ins_ --grid small --calib raw > night/decode_ins.log 2>&1; log decode
cd /workspace/koz3/wt/w4-pix; T=/workspace/koz3/w4geo/train; export EXPDS_METRIC=$PWD/materials/compute_metrics.py; mkdir -p $W/c_ins
for tg in ins_fs1_n0.75_p0.01_raw_x1 ins_fs1_n1_p0.01_raw_x1; do
  $PY scripts/pix_clean_oof.py compose --root $T --config configs/v3_on.toml --cache $W/cache --tag $tg --set 'geophys.pixgbm.impl="clean"' --set geophys.pixgbm.replace=true --set 'geophys.pixgbm.maps=["maggr"]' --set 'geophys.pixgbm.classes=["finds_points"]' --out $W/c_ins/C_S_$tg.geojson > $W/c_ins/compose_$tg.log 2>&1 &
done; wait; log compose
cd $W
for tg in ins_fs1_n0.75_p0.01_raw_x1 ins_fs1_n1_p0.01_raw_x1; do
  bash m910/chain.sh ${tg} c_ins/C_S_$tg.geojson > night/chain_$tg.log 2>&1 && \
  $PY sweep/merge_finds.py cstar/C_CetO3_Sr8G_noTerFl_L_E1.geojson m910/C_${tg}.geojson night/E1_${tg}.geojson >> night/chain_$tg.log 2>&1; log "chain $tg rc=$?"
done
$PY night/mkM_any.py night/E1_ins_fs1_n1_p0.01_raw_x1.geojson cstar/C_CetO3_Sr8G_noTerFl_L_E1_INS.geojson 1 1 2 1 >> night/chain_ins.log 2>&1
$PY night/mkM_any.py night/E1_ins_fs1_n1_p0.01_raw_x1.geojson cstar/C_CetO3_Sr8G_noTerFl_L_E1_INSM3.geojson 0.9 0.5 2.0 0.7 >> night/chain_ins.log 2>&1
bash night/gate_eval.sh INS INS INSM3; log "gate INS done"
