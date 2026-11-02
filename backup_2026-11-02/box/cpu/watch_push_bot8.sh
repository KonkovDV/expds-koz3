#!/bin/bash
# Polls acc1 every 10 min; when <2 GPU kernels running, pushes bot8 OOF honest10 with GPU once.
K=~/.local/bin/kaggle; v=KAGGLE_KEY_1
for i in $(seq 1 120); do
  out=$(env -u KAGGLE_USERNAME -u KAGGLE_KEY KAGGLE_API_TOKEN="${!v}" $K kernels push -p /workspace/koz3/pull/koz3-bot8-oof-honest10 2>&1 | tail -1)
  echo "$(date '+%F %T MSK') $out"
  case "$out" in *"Maximum batch GPU"*) sleep 600;; *) echo PUSHED_OR_ERROR; exit 0;; esac
done
