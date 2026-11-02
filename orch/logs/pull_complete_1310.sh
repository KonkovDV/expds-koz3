#!/bin/bash
cd /workspace/koz3
P=orch/pool_kaggle.sh
pull(){ mkdir -p "$2"; echo "$(date +%H:%M) pull $1 -> $2"; timeout 3600 nice -n 10 $P ${1%%/*} kernels output $1 -p "$2" -q >/dev/null 2>&1; echo "$(date +%H:%M) rc=$? $1 files=$(find $2 -type f | wc -l) size=$(du -sh $2 | cut -f1)"; }
pull elenakonkova/koz3-w1-dfine-or-final out/w1-dfine-or-final
pull elenakonkova/koz3-w5-cloudfeat out/w5-cloudfeat
pull dmitrijkv/koz3-w2-hillshade-parity-s3 w2hs/out
pull elenakonkova/koz3-w2-lihm-c3sc-f01234 w2/oof/lihm/koz3-w2-lihm-c3sc-f01234
pull elenakonkova/koz3-w2-cnx-c3sc-s3f4 out/w2-cnx-c3sc-s3f4
echo DONE
