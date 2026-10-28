#!/bin/bash
# Kaggle (2×T4): RF-DETR-M/L без COCO-чекпойнта (энкодер DINOv2-S, Apache-2.0) по 5 семейным фолдам, две полосы по GPU.
# Только честный замер — в сдачу ничего до решения по весам/письма организаторов.
# usage: bash kaggle/rfdetr_folds.sh <DET_ROOT> <OUT_ROOT> [MODEL] [ITERS] [DINO_DIR]
#   DET_ROOT: f0..f4 и grid_f0..grid_f4 из coco_from_cache.py (как для D-FINE); DINO_DIR — локальная facebook/dinov2-small (иначе HF)
set -euo pipefail
DET=$1; OUT=$2; MODEL=${3:-rfdetr-m}; ITERS=${4:-8000}; DINO=${5:-}
HERE=$(cd "$(dirname "$0")/.." && pwd)
DARG=(); [ -n "$DINO" ] && DARG=(--dinov2-dir "$DINO")
lane() {  # $1 GPU, остальное — фолды
  local g=$1; shift
  for k in "$@"; do
    CUDA_VISIBLE_DEVICES=$g python "$HERE/scripts/rfdetr_train.py" --data "$DET/f$k" --out "$OUT/f$k" "${DARG[@]}" \
      --model "$MODEL" --size 640 --iters "$ITERS" --bs 4 --accum 4 --fp16 --loss mal --cp-prob 0.5 \
      --tta-flip --log-every 200 > "$OUT/f$k.train.log" 2>&1
    CUDA_VISIBLE_DEVICES=$g python "$HERE/scripts/rfdetr_train.py" --data "$DET/f$k" --out "$OUT/f$k/grid" \
      --load "$OUT/f$k/model" --pred-data "$DET/grid_f$k" --pred-split valid --tta-flip --pred-bs 16 --fp16 \
      > "$OUT/f$k.grid.log" 2>&1
  done
}
mkdir -p "$OUT"
lane 0 0 2 4 & lane 1 1 3 & wait
echo done; ls "$OUT"/f*/grid/preds_valid.json
