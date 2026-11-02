#!/bin/bash
# Build GRID tile cache (expds.tiles --grid-overlap 0.25, sp_or kurgans, res 0.5, tile 512) per pinned fold (manifest).
# Out: /workspace/koz3/out/sat2-f0/grid/f{k}/  (CPU only)
set -e
K=/workspace/koz3; CODE=$K/wt/sp-large-f0; OUT=$K/out/sat2-f0/grid; mkdir -p $OUT
source $K/venv_sat2/bin/activate
cd $CODE
for k in "$@"; do
  regs=$(python3 -c "import json;d=json.load(open('$K/orch/bot-W1/sat2_dataset_manifest.json'));print(' '.join(r['region'] for r in d['regions'] if r['fold']==$k and r['tifs']))")
  python -m expds.tiles --root $K/w5data/sat_raw --gt $K/w5data/gt_train_v2.geojson --markup sp_or --out $OUT/f$k \
    --res 0.5 --tile 512 --grid-overlap 0.25 --classes kurgany_povrezhdennye kurgany_tselye --regions $regs > $OUT/f$k.log 2>&1
  echo "fold $k done $(date +%T)"
done
