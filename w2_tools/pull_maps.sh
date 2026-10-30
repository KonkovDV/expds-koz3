#!/usr/bin/env bash
# W2: скачать maps_cnx_*.npz c3-sc s0/s1 (только чтение)
K(){ local acc=$1; shift
  if [ "$acc" = elenakonkova ]; then env -u KAGGLE_USERNAME -u KAGGLE_KEY KAGGLE_API_TOKEN="$KAGGLE_KEY_1" ~/.local/bin/kaggle "$@"
  else /workspace/koz3/orch/pool_kaggle.sh "$acc" "$@"; fi; }
for it in s0:dmitrijkv/koz3-cnx-li-c3sc-f01 s0:dmitrijkv/koz3-cnx-li-c3sc-f23 s0:dmitrijkv/koz3-cnx-li-c3sc-f4-v2 s1:dmitrijkv/koz3-cnx-li-c3sc-s1f01 s1:elenakonkova/koz3-cnx-li-c3sc-s1f23 s1:elenakonkova/koz3-cnx-li-c3sc-s1f4; do
  seed=${it%%:*}; k=${it#*:}; acc=${k%%/*}; d=/workspace/koz3/w2/maps/$seed/${k#*/}; mkdir -p $d
  nice -n 10 timeout 3000 bash -c "$(declare -f K); K $acc kernels output $k -p $d --file-pattern 'maps_cnx_.*\.npz$' -q -o" >/dev/null 2>&1; echo "$k $? $(find $d -name '*.npz' | wc -l) $(du -sh $d | cut -f1)"
done
