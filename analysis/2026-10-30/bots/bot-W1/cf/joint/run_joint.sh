#!/bin/bash
# poll box/q1|q2 lease, run joint gate (nice 10, 4 threads), release
cd /workspace/koz3/w1/cf/joint
L=/workspace/koz3/orch/lease.sh; SLOT=""
while [ -z "$SLOT" ]; do
  for s in slot:box/q1 slot:box/q2; do
    if $L acquire $s bot-W1-aeor 2 --note "W1 joint gate RR1s-cf+M_rp on v3_on (PREREG_W1 16:10), nice10 thr4" >/dev/null 2>&1; then SLOT=$s; break; fi
  done
  [ -z "$SLOT" ] && sleep 30
done
echo "$(date +%H:%M:%S) got $SLOT"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 PYTHONPATH=/workspace/koz3/wt/w1
nice -n 10 /workspace/koz3/venv/bin/python joint_gate.py 2000 /workspace/koz3/w1/cf/joint/joint_gate.json > joint_gate.log 2>&1
echo "$(date +%H:%M:%S) exit $?"
$L release $SLOT bot-W1-aeor
echo "$(date +%H:%M:%S) released $SLOT"
