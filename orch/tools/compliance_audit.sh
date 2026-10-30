#!/usr/bin/env bash
# Read-only аудит соответствия (MASTER_PLAN §Compliance). Без checkout, без сети, без токенов.
# usage: compliance_audit.sh [ref ...]   (по умолчанию origin/main origin/bots/w4-repro-v2)
set -u
R=/workspace/koz3/repo; OUT=/workspace/koz3/orch/logs; mkdir -p "$OUT"
refs=("$@"); [ ${#refs[@]} -eq 0 ] && refs=(origin/main origin/bots/w4-repro-v2)
for ref in "${refs[@]}"; do
  git -C "$R" rev-parse -q --verify "$ref" >/dev/null || { echo "нет ref $ref"; continue; }
  f="$OUT/compliance_audit_$(echo "$ref" | tr '/' '_').md"
  {
    echo "# compliance audit $ref @ $(git -C "$R" rev-parse --short "$ref") — $(date '+%d.%m %H:%M') MSK"
    echo "## 1. Следы EN (EhimenNathan/EXpedition/maratgaliulin) и geo_pixgbm"
    git -C "$R" grep -n -I -i -E "ehimen|expedition-?nathan|EXpedition|maratgaliulin|geo_pixgbm" "$ref" -- . ':!orch' ':!docs' | cut -c1-200 | head -60
    echo "## 2. Запрещённые зависимости (ultralytics/YOLO/SAM3/DINOv3)"
    git -C "$R" grep -n -I -i -E "import ultralytics|from ultralytics|sam3|dinov3|yolov?[0-9]" "$ref" -- inference expds scripts requirements*.txt pyproject.toml | cut -c1-200 | head -40
    echo "## 3. Данные в дереве инференса (tif/las/laz/geojson/gpkg)"
    git -C "$R" ls-tree -r -l "$ref" -- inference models | awk '$5 ~ /\.(tif|tiff|las|laz|geojson|gpkg|shp)$/ {print $4, $5}' | head -40
    echo "## 4. Обязательные файлы ПКК"
    for p in metadata.json LICENSE README.md ai_declaration.md requirements.lock; do
      git -C "$R" cat-file -e "$ref:$p" 2>/dev/null && echo "ok  $p" || echo "НЕТ $p"; done
    echo "## 5. Размер inference/ + models/ (байт)"
    git -C "$R" ls-tree -r -l "$ref" -- inference models | awk '{s+=$4} END {print s+0}'
  } > "$f" 2>&1
  echo "→ $f"
done
