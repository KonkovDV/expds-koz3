#!/usr/bin/env bash
# W2 [li_cnx]: собрать OOF ConvNeXt-ядер после COMPLETE (только чтение Kaggle; ничего не пушит).
# usage: collect_cnx.sh [--maps]   (по умолчанию без ckpt/weights/maps)
set -uo pipefail
OUT=/workspace/koz3/w2/oof; mkdir -p "$OUT"
PAT='(oof_cnx_.*\.geojson|gt_li_cnx_.*\.geojson|region_fold\.json|report_cnx_full\.json|.*\.log)$'
[ "${1:-}" = "--maps" ] && PAT='(oof_cnx_.*\.geojson|gt_li_cnx_.*\.geojson|region_fold\.json|report_cnx_full\.json|maps_cnx_.*\.npz|.*\.log)$'
K(){ local acc=$1; shift
  if [ "$acc" = elenakonkova ]; then env -u KAGGLE_USERNAME -u KAGGLE_KEY KAGGLE_API_TOKEN="$KAGGLE_KEY_1" ~/.local/bin/kaggle "$@"
  else /workspace/koz3/orch/pool_kaggle.sh "$acc" "$@"; fi; }
# seed : kernel
LIST="s0:dmitrijkv/koz3-cnx-li-c3sc-f01 s0:dmitrijkv/koz3-cnx-li-c3sc-f23 s0:dmitrijkv/koz3-cnx-li-c3sc-f4-v2
s1:dmitrijkv/koz3-cnx-li-c3sc-s1f01 s1:elenakonkova/koz3-cnx-li-c3sc-s1f23 s1:elenakonkova/koz3-cnx-li-c3sc-s1f4
s0:ACC/koz3-w2-cnx-c5sc-f01 s0:ACC/koz3-w2-cnx-c5sc-f23 s0:ACC/koz3-w2-cnx-c5sc-f4"
for it in $LIST; do seed=${it%%:*}; ref=${it#*:}
  for acc in $( [ "${ref%%/*}" = ACC ] && echo "elenakonkova dmitrijkv" || echo "${ref%%/*}" ); do
    k="$acc/${ref#*/}"; st=$(K "$acc" kernels status "$k" 2>&1 | tail -1)
    case "$st" in *COMPLETE*) ;; *) echo "skip $k: ${st##* }"; continue;; esac
    d="$OUT/$seed/${ref#*/}"; mkdir -p "$d"
    timeout 1800 nice -n 10 bash -c "$(declare -f K); K $acc kernels output $k -p '$d' --file-pattern '$PAT' -q -o" >/dev/null 2>&1 \
      && echo "ok $k → $d" || echo "FAIL download $k"
  done
done
nice -n 10 python3 /workspace/koz3/w2/tools/merge_cnx.py "$OUT"
