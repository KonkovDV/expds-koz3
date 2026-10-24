#!/usr/bin/env bash
# Прогон архива в условиях платформы (Linux/WSL): без сети, код только для чтения,
# пользователь не root, без записи в /etc/passwd.
# На площадке 96 ГБ. На этой машине по умолчанию 8 ГБ: MEM=96g для полной имитации.
#
#   bash scripts/platform_sim.sh pantum/n7q:20261024 /data/mock_val /tmp/platform_sim dist/submission.zip
set -euo pipefail
IMAGE="$1"; DATA="$2"; OUT="$3"; ZIP="${4:-dist/submission.zip}"
MEM="${MEM:-8g}"
CODE="$(mktemp -d)"
unzip -q "$ZIP" -d "$CODE"
mkdir -p "$OUT" && chmod 777 "$OUT" && rm -f "$OUT/result.geojson"
GPU=(--gpus all)
[ "${NO_GPU:-0}" = "1" ] && GPU=()
start=$(date +%s)
docker run --rm "${GPU[@]}" --network none --read-only --user 12345:12345 \
  --memory "$MEM" --pids-limit 8192 --shm-size 16g \
  -v "$CODE:/app:ro" -v "$DATA:/data/input:ro" -v "$OUT:/data/output" -w /app \
  "$IMAGE" python inference/main.py /data/input /data/output/result.geojson 2>&1 | tee "$OUT/run.log"
echo "время контейнера: $(( $(date +%s) - start )) с"
python -m expds.writer check "$OUT/result.geojson"
ls -A "$OUT" | grep -v -x -e result.geojson -e run.log && echo "ВНИМАНИЕ: в папке результата остались файлы" || true
grep -E "среда:|опись входа|ERROR|Traceback|Fatal Python error|нет файлов моделей|без CRS" "$OUT/run.log" || true
rm -rf "$CODE"
