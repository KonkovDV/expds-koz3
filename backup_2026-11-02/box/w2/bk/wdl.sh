#!/usr/bin/env bash
# скачивание недостающих весов f0/f1 (s0,s1,s2) из output ядер, read-only
set -uo pipefail
PK=/workspace/koz3/orch/pool_kaggle.sh
for it in s0:dmitrijkv/koz3-cnx-li-c3sc-f01 s1:dmitrijkv/koz3-cnx-li-c3sc-s1f01 s2:dmitrijkv/koz3-w2-cnx-c3sc-s2f01; do
  seed=${it%%:*}; k=${it#*:}; d=/workspace/koz3/w2/bk/wdl/$seed/${k#*/}; mkdir -p $d
  timeout 1800 nice -n 10 $PK dmitrijkv kernels output $k -p "$d" --file-pattern '^weights_cnx_c3-sc_f[01]\.pt$' -q -o >/dev/null 2>&1 && echo "ok $k" || echo "FAIL $k"
  ls -la $d
done
