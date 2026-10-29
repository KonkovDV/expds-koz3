#!/bin/bash
# Kaggle (2×T4): D-FINE по 5 семейным фолдам, две полосы по GPU. Только честный замер — в сдачу ничего до письма организаторов.
# usage: bash kaggle/dfine_folds.sh <DET_ROOT> <OUT_ROOT> [MODEL] [ITERS]
#   DET_ROOT: f0..f4 (coco_from_cache --layout rfdetr --folds 5 --fold k) и grid_f0..grid_f4 (сплошная сетка, тот же --fold k)
set -euo pipefail
DET=$1; OUT=$2; MODEL=${3:-dfine-l}; ITERS=${4:-8000}
HERE=$(cd "$(dirname "$0")/.." && pwd)
lane() {  # $1 GPU, остальное — фолды
  local g=$1; shift
  for k in "$@"; do
    CUDA_VISIBLE_DEVICES=$g python "$HERE/scripts/det_train_hf.py" --data "$DET/f$k" --out "$OUT/f$k" ${EXTRA_ARGS:-} \
      --model "$MODEL" --iters "$ITERS" --bs 8 --size 640 --fp16 --workers 2 --tta-flip --log-every 200 \
      > "$OUT/f$k.train.log" 2>&1
    CUDA_VISIBLE_DEVICES=$g python "$HERE/scripts/det_train_hf.py" --data "$DET/f$k" --out "$OUT/f$k/grid" \
      --load "$OUT/f$k/model" --pred-data "$DET/grid_f$k" --pred-split valid --size 640 --tta-flip --pred-bs 16 \
      > "$OUT/f$k.grid.log" 2>&1
  done
}
mkdir -p "$OUT"
lane 0 0 2 4 & lane 1 1 3 & wait
echo done; ls "$OUT"/f*/grid/preds_valid.json
