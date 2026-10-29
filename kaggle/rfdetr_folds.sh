#!/bin/bash
# Kaggle (2×T4): RF-DETR-M/L по 5 семейным фолдам, две полосы по GPU. По умолчанию INIT=coco — дообучение чекпойнта
# RF-DETR COCO (rf-detr-medium.pth, Apache-2.0; веса COCO разрешены организаторами). Честный замер OOF.
# usage: [INIT=coco|dinov2|scratch] [COCO_W=/path/rf-detr-medium.pth] [SIZE=640] bash kaggle/rfdetr_folds.sh <DET_ROOT> <OUT_ROOT> [MODEL] [ITERS] [DINO_DIR]
#   DET_ROOT: f0..f4 и grid_f0..grid_f4 из coco_from_cache.py (как для D-FINE)
#   COCO_W: локальный чекпойнт (иначе скачивается в $RF_HOME, md5 проверяет rfdetr); DINO_DIR нужен только при INIT=dinov2
set -euo pipefail
DET=$1; OUT=$2; MODEL=${3:-rfdetr-m}; ITERS=${4:-8000}; DINO=${5:-}
HERE=$(cd "$(dirname "$0")/.." && pwd)
INIT=${INIT:-coco}; COCO_W=${COCO_W:-}; SIZE=${SIZE:-640}  # rfdetr-b (O365): SIZE кратно 56, напр. 672
DARG=(--init "$INIT"); [ -n "$DINO" ] && DARG+=(--dinov2-dir "$DINO"); [ -n "$COCO_W" ] && DARG+=(--coco-weights "$COCO_W")
lane() {  # $1 GPU, остальное — фолды
  local g=$1; shift
  for k in "$@"; do
    CUDA_VISIBLE_DEVICES=$g python "$HERE/scripts/rfdetr_train.py" --data "$DET/f$k" --out "$OUT/f$k" "${DARG[@]}" \
      --model "$MODEL" --size "$SIZE" --iters "$ITERS" --bs 4 --accum 4 --fp16 --loss mal --cp-prob 0.5 \
      --tta-flip --log-every 200 > "$OUT/f$k.train.log" 2>&1
    CUDA_VISIBLE_DEVICES=$g python "$HERE/scripts/rfdetr_train.py" --data "$DET/f$k" --out "$OUT/f$k/grid" \
      --load "$OUT/f$k/model" --pred-data "$DET/grid_f$k" --pred-split valid --tta-flip --pred-bs 16 --fp16 \
      > "$OUT/f$k.grid.log" 2>&1
  done
}
mkdir -p "$OUT"
lane 0 0 2 4 & lane 1 1 3 & wait
echo done; ls "$OUT"/f*/grid/preds_valid.json
