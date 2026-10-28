#!/usr/bin/env bash
# Обёртка: собрать архив эксперимента и сразу проверить (bot-8).
#   ./build_submission.sh base                         # HEAD репозитория
#   ./build_submission.sh thr2 --config my.toml --weight models/li_r18_hard2.pt=/x/li.pt
# Переменные: REPO (по умолчанию ./repo), PY (./venv/bin/python), CHECK_ARGS (например "--no-smoke").
set -euo pipefail
B8="$(cd "$(dirname "$0")" && pwd)"
REPO="${REPO:-$B8/repo}"; PY="${PY:-$B8/venv/bin/python}"
label="$1"; shift
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 RAYON_NUM_THREADS=2
exec nice -n 10 "$PY" "$B8/build_submission.py" --repo "$REPO" --out-dir "$B8/dist" --label "$label" "$@" \
    --check --check-args "--python $PY ${CHECK_ARGS:-}"
