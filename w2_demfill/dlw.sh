#!/usr/bin/env bash
set -uo pipefail
K(){ local acc=$1; shift
  if [ "$acc" = elenakonkova ]; then env -u KAGGLE_USERNAME -u KAGGLE_KEY KAGGLE_API_TOKEN="$KAGGLE_KEY_1" ~/.local/bin/kaggle "$@"
  else /workspace/koz3/orch/pool_kaggle.sh "$acc" "$@"; fi; }
for it in s0:dmitrijkv/koz3-cnx-li-c3sc-f23 s0:dmitrijkv/koz3-cnx-li-c3sc-f4-v2 s1:elenakonkova/koz3-cnx-li-c3sc-s1f23 s1:elenakonkova/koz3-cnx-li-c3sc-s1f4; do
  seed=${it%%:*}; k=${it#*:}; acc=${k%%/*}; d=/workspace/koz3/w2/demfill/w/$seed/${k#*/}; mkdir -p $d
  timeout 2400 nice -n 10 bash -c "$(declare -f K); K $acc kernels output $k -p '$d' --file-pattern '^weights_cnx_c3-sc_f[234]\.pt$' -q -o" >/dev/null 2>&1 && echo "ok $k" || echo "FAIL $k"
  ls -la $d
done
