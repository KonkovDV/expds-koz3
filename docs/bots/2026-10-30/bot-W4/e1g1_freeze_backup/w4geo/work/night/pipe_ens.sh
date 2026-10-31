#!/bin/bash
# usage: pipe_ens.sh NAME probsdir...   (mean maps -> decode -> compose S -> chain (Gtype_n300+merge+loops) -> E1 base merge)
set -u; N=$1; shift; W=/workspace/koz3/w4geo/work; cd $W; PY="nice -n 10 $W/../venv/bin/python"
log(){ echo "$(date '+%H:%M:%S') [$N] $*" | tee -a night/pipe.log; }
$PY night/mkens.py probs_$N "$@" > night/mkens_$N.log 2>&1; log mkens
$PY sweep/finds_sweep.py decode --probs probs_$N --prefix ${N}_ --grid ${GRID:-small} --calib raw > night/decode_$N.log 2>&1; log decode
cd /workspace/koz3/wt/w4-pix; T=/workspace/koz3/w4geo/train; export EXPDS_METRIC=$PWD/materials/compute_metrics.py; mkdir -p $W/c_$N
for tg in $(cat $W/sweep/tags_${N}_${GRID:-small}.txt); do
  $PY scripts/pix_clean_oof.py compose --root $T --config configs/v3_on.toml --cache $W/cache --tag $tg --set 'geophys.pixgbm.impl="clean"' --set geophys.pixgbm.replace=true --set 'geophys.pixgbm.maps=["maggr"]' --set 'geophys.pixgbm.classes=["finds_points"]' --out $W/c_$N/C_S_$tg.geojson > $W/c_$N/compose_$tg.log 2>&1 &
done; wait; log compose
cd $W
for tg in $(cat sweep/tags_${N}_${GRID:-small}.txt); do
  bash m910/chain.sh ${tg} c_$N/C_S_$tg.geojson > night/chain_$tg.log 2>&1 && \
  $PY sweep/merge_finds.py cstar/C_CetO3_Sr8G_noTerFl_L_E1.geojson m910/C_${tg}.geojson night/E1_${tg}.geojson >> night/chain_$tg.log 2>&1; log "chain $tg rc=$?"
done
