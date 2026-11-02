#!/bin/bash
# D: frozen E1_G1 recipe (verified byte-identical: decode fs3_n0.5_p0.02_raw_x1 -> compose S -> chain -> merge E1 -> mkM_any .9 .25 1.75 .6)
# applied to alt prob maps: NAME in {bens5 (base s0..s4 mean), e6b5 (mean of ens6m & bens5)}. Gate raw+_ht vs REF + honest eval incl frozen G1. No LB.
set -u; W=/workspace/koz3/w4geo/work; cd $W; PY="nice -n 10 $W/../venv/bin/python"; O=/workspace/koz3/orch/bot-W3/geo5_seedens
export EXPDS_METRIC=/workspace/koz3/wt/w4-pix/materials/compute_metrics.py
log(){ echo "$(date '+%H:%M:%S') [g1] $*" | tee -a $O/run.log; }
while pgrep -f "dump --models models_bs4" >/dev/null; do sleep 20; done; [ $(ls probs_bs4/*/*.npz | wc -l) -eq 19 ] || { log "bs4 incomplete"; exit 1; }; log "dump s4 done"
[ -d probs_bens5 ] || $PY night/mkens.py probs_bens5 probs probs_bs1 probs_bs2 probs_bs3 probs_bs4 > $O/mkens_bens5.log 2>&1
while pgrep -f "mkens.py probs_bens5" >/dev/null; do sleep 10; done
$PY night/mkens.py probs_e6b5 probs_ens6m probs_bens5 > $O/mkens_e6b5.log 2>&1; log "mkens rc=$?"
TG=fs3_n0.5_p0.02_raw_x1; L=()
for N in bens5 e6b5; do
  $PY sweep/finds_sweep.py decode --probs probs_$N --prefix g1${N}_ --grid main --calib raw > $O/decode_$N.log 2>&1; log "decode $N rc=$?"
  t=g1${N}_$TG
  (cd /workspace/koz3/wt/w4-pix && $PY scripts/pix_clean_oof.py compose --root /workspace/koz3/w4geo/train --config configs/v3_on.toml --cache $W/cache --tag $t --set 'geophys.pixgbm.impl="clean"' --set geophys.pixgbm.replace=true --set 'geophys.pixgbm.maps=["maggr"]' --set 'geophys.pixgbm.classes=["finds_points"]' --out $O/C_S_$t.geojson > $O/compose_$N.log 2>&1); log "compose $N rc=$?"
  bash m910/chain.sh $t $O/C_S_$t.geojson > $O/chain_$N.log 2>&1 && $PY sweep/merge_finds.py cstar/C_CetO3_Sr8G_noTerFl_L_E1.geojson m910/C_$t.geojson $O/E1_$t.geojson >> $O/chain_$N.log 2>&1; log "chain $N rc=$?"
  NM=CetO3_Sr8G_noTerFl_L_E1_G1R_$N
  $PY night/mkM_any.py $O/E1_$t.geojson cstar/C_$NM.geojson 0.9 0.25 1.75 0.6 > $O/mkM_$N.log 2>&1; log "mkM $N rc=$?"
  bash sweep/gate_cstar.sh $NM; log "gate $N rc=$?"; L+=($W/cstar/C_$NM.geojson $W/cstar/C_${NM}_ht.geojson)
done
cd /workspace/koz3/wt/w4-pix; $PY scripts/pix_clean_eval.py --gt /workspace/koz3/w5data/gt_train_v2.geojson --a $W/A18.geojson --c $W/cstar/C_CetO3_Sr8G_noTerFl_L_E1_G1.geojson $W/cstar/C_CetO3_Sr8G_noTerFl_L_E1_G1_ht.geojson "${L[@]}" --out $O/R_G1R.json > $O/R_G1R.log 2>&1; log "eval rc=$?"
log DONE
