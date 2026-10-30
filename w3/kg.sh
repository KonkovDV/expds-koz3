#!/bin/bash
# W3 helper: kg.sh <account> <kaggle args>; elenakonkova -> $KAGGLE_KEY_1, others via pool_kaggle.sh. Token never printed.
u="$1"; shift
if [[ "$u" == elenakonkova ]]; then env -u KAGGLE_USERNAME -u KAGGLE_KEY KAGGLE_API_TOKEN="$KAGGLE_KEY_1" ~/.local/bin/kaggle "$@"
else /workspace/koz3/orch/pool_kaggle.sh "$u" "$@"; fi
