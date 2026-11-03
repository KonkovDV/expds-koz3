#!/bin/bash
# usage: orch/pool_kaggle.sh <username> <kaggle args...>
# NOTE (30.10 box): env labels were swapped vs actual tokens — KEY_1→dmitrijkv, KEY_2→elenakonkova
u="$1"; shift
if [[ "$u" == elenakonkova ]]; then exec env -u KAGGLE_USERNAME -u KAGGLE_KEY KAGGLE_API_TOKEN="$KAGGLE_KEY_2" ~/.local/bin/kaggle "$@"; fi
if [[ "$u" == dmitrijkv ]]; then exec env -u KAGGLE_USERNAME -u KAGGLE_KEY KAGGLE_API_TOKEN="$KAGGLE_KEY_1" ~/.local/bin/kaggle "$@"; fi
t=$(python3 -c "
import json,sys
d=json.load(open('/workspace/koz3/local_ctx/ctx/kaggle_pool_4x2_t4.json'));u=sys.argv[1]
print(next(n['token'] for n in d['nodes'] if n['username']==u) if 'nodes' in d else d[u].get('key') or d[u].get('token'))" "$u")
if [[ "$t" == KGAT_* ]]; then env -u KAGGLE_USERNAME -u KAGGLE_KEY KAGGLE_API_TOKEN="$t" ~/.local/bin/kaggle "$@"
else env -u KAGGLE_API_TOKEN KAGGLE_USERNAME="$u" KAGGLE_KEY="$t" ~/.local/bin/kaggle "$@"; fi
