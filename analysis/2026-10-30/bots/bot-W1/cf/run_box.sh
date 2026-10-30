#!/bin/bash
# waits for free slot:box/q1|q2, runs cloudfeat (S3 keys only in env) + PREREG 09:05 gate, releases slot
cd /workspace/koz3
while true; do
  for s in q1 q2; do
    if orch/lease.sh acquire slot:box/$s bot-W1-aeor 5 --note "CI2-1 cloudfeat+gate on box (bot1 option b), nice10 thr4" 2>&1 | grep -q '^OK'; then SLOT=$s; break 2; fi
  done
  sleep 10
done
echo "$(date '+%H:%M') got slot:box/$SLOT"
( set -a; . /workspace/koz3/local_ctx/s3_train.env; set +a
  export OMP_NUM_THREADS=4 CF_IN=/workspace/koz3/w1/cf/ds CF_OUT=/workspace/koz3/w1/cf/box_out CF_TMP=/workspace/koz3/w1/cf/tmp_box DEADLINE_H=6
  nice -n 10 /workspace/koz3/w1/venv/bin/python w1/cf/kernel/koz3-w5-cloudfeat.py ) > w1/cf/box_feat.log 2>&1
echo "$(date '+%H:%M') feats exit $?"; rm -rf w1/cf/tmp_box
cd w1/cf; export W5_ROOT=/workspace/koz3/wt/w5 W5_DATA=/workspace/koz3/w1/cf/w5d OMP_NUM_THREADS=4
nice -n 10 python3 build_x4.py box_out/cloudfeat.csv && \
nice -n 10 /workspace/koz3/venv/bin/python nested2.py --xset X4 --tag w5cf --label first --threads 4 > nested_w5cf.log 2>&1 && echo "$(date '+%H:%M') nested ok" && \
RR_TAG=w5cf nice -n 10 /workspace/koz3/venv/bin/python ws_rr.py > ws_rr_w5cf.log 2>&1 && echo "$(date '+%H:%M') ws_rr ok" && \
nice -n 10 /workspace/koz3/venv/bin/python gate_cf.py > gate_cf.log 2>&1; echo "$(date '+%H:%M') gate exit $?"
cd /workspace/koz3; orch/lease.sh release slot:box/$SLOT bot-W1-aeor; echo "$(date '+%H:%M') released slot:box/$SLOT"
