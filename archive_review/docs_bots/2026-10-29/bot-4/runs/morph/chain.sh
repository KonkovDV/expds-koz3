#!/bin/bash
# по одной раскладке: k10 (продолжение с чекпойнта), lofo, seed0..seed4; повтор после OOM-kill с чекпойнта
cd /workspace/koz3_shared/bot-4/work/morph
for L in k10 lofo seed0 seed1 seed2 seed3 seed4; do
  [ -f res/res_$L.json ] && continue
  for try in 1 2 3 4 5; do
    ./run.sh $L $L.r$try
    tail -1 res/log_$L.r$try.txt | grep -q "EXIT 0" && break
    sleep 60
  done
done
echo "CHAIN DONE $(date '+%F %T')" > res/chain.done
