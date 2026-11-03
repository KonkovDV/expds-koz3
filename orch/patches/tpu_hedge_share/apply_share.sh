#!/bin/bash
# NOT RUN. Requires Elena's explicit OK. Adds dmitrijkv as READER (datasets stay private).
set -e; cd /workspace/koz3
for d in koz3-w5-cands koz3-cnx-li-prep koz3-s3-cred; do
  orch/pool_kaggle.sh elenakonkova datasets metadata elenakonkova/$d --update -p orch/patches/tpu_hedge_share/$d
done
for d in koz3-w5-cands koz3-cnx-li-prep koz3-s3-cred; do
  echo "== $d as dmitrijkv"; orch/pool_kaggle.sh dmitrijkv datasets files elenakonkova/$d 2>&1 | head -3
done
