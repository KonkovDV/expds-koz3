#!/bin/bash
# W4: опрос kernel koz3-w4-v3onchk v5 не чаще раза в 30 мин; по завершении — вывод в out_v3onfix2
for i in $(seq 1 6); do
  s=$(bash /workspace/koz3/orch/pool_kaggle.sh aaaa0000004 kernels status aaaa0000004/koz3-w4-v3onchk 2>&1 | tail -1)
  echo "$(date '+%d.%m %H:%M MSK') $s"
  if [[ "$s" == *COMPLETE* || "$s" == *ERROR* || "$s" == *CANCEL* ]]; then
    mkdir -p /workspace/koz3/w4geo/kaggle/out_v3onfix2
    bash /workspace/koz3/orch/pool_kaggle.sh aaaa0000004 kernels output aaaa0000004/koz3-w4-v3onchk -p /workspace/koz3/w4geo/kaggle/out_v3onfix2 2>&1 | tail -2
    echo DONE; exit 0
  fi
  sleep 1800
done
