#!/usr/bin/env bash
# Собрать архив из git-коммита и сразу проверить (bot-8).
#   scripts/build_submission_git.sh base                       # HEAD репозитория
#   scripts/build_submission_git.sh thr2 --config my.toml --weight models/li_r18_hard2.pt=/x/li.pt
#   REF=bot8-reliability-on-bot7 scripts/build_submission_git.sh rel
# Переменные: REPO (корень этого репо), PY (python3), OUT (./dist), REF (HEAD), CHECK_ARGS (например "--no-smoke").
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="${REPO:-$(git -C "$HERE" rev-parse --show-toplevel)}"; PY="${PY:-python3}"
label="$1"; shift
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 RAYON_NUM_THREADS=2
exec nice -n 10 "$PY" "$HERE/build_submission_git.py" --repo "$REPO" --ref "${REF:-HEAD}" --out-dir "${OUT:-$PWD/dist}" \
    --label "$label" "$@" --check --check-args "--python $PY ${CHECK_ARGS:-}"
