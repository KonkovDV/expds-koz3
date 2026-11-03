#!/bin/bash
# D: seed-ens base finds (s0..s4) -> pipe_ens (mean maps, decode small, compose S, chain, E1 merge) -> gate vs A18 incl frozen E1/E1_G1. No LB. nice10, 3 thr.
set -u; W=/workspace/koz3/w4geo/work; cd $W; PY="nice -n 10 $W/../venv/bin/python"; O=/workspace/koz3/orch/bot-W3/geo5_seedens
log(){ echo "$(date '+%H:%M:%S') $*" | tee -a $O/run.log; }
for s in 1 2 3 4; do $PY sweep/finds_sweep.py dump --models models_bs$s --probs probs_bs$s --threads 3 > $O/dump_bs$s.log 2>&1; log "dump s$s rc=$? n=$(ls probs_bs$s/*/*.npz | wc -l)"; done
GRID=small bash night/pipe_ens.sh bens5 probs probs_bs1 probs_bs2 probs_bs3 probs_bs4; log "pipe_ens rc=$?"
L=(); for tg in $(cat sweep/tags_bens5_small.txt); do cp night/E1_${tg}.geojson cstar/C_CetO3_Sr8G_noTerFl_L_E1_BENS5_${tg}.geojson && L+=($W/cstar/C_CetO3_Sr8G_noTerFl_L_E1_BENS5_${tg}.geojson); done
cd /workspace/koz3/wt/w4-pix; export EXPDS_METRIC=$PWD/materials/compute_metrics.py
$PY scripts/pix_clean_eval.py --gt /workspace/koz3/w5data/gt_train_v2.geojson --a $W/A18.geojson --c $W/cstar/C_CetO3_Sr8G_noTerFl_L_E1.geojson $W/cstar/C_CetO3_Sr8G_noTerFl_L_E1_G1.geojson "${L[@]}" --out $O/R_BENS5.json > $O/R_BENS5.log 2>&1; log "eval rc=$?"
log DONE
