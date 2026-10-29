#!/bin/bash
# usage: orch/pool_kaggle.sh <username> <kaggle args...>  — токен читается из файла, не печатается
u="$1"; shift
t=$(python3 -c "import json,sys;print(next(n['token'] for n in json.load(open('/workspace/koz3/local_ctx/ctx/kaggle_pool_4x2_t4.json'))['nodes'] if n['username']==sys.argv[1]))" "$u")
if [[ "$t" == KGAT_* ]]; then env -u KAGGLE_USERNAME -u KAGGLE_KEY KAGGLE_API_TOKEN="$t" ~/.local/bin/kaggle "$@"
else env -u KAGGLE_API_TOKEN KAGGLE_USERNAME="$u" KAGGLE_KEY="$t" ~/.local/bin/kaggle "$@"; fi
