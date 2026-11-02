#!/bin/bash
while kill -0 20871 2>/dev/null; do sleep 120; done
cd /workspace/koz3/cpuk/box_s41
echo "- $(date '+%d.%m %H:%M MSK') box-cpu: PID 20871 done → start kurgan seeds 41,42 (LOCAL_TEST, nice 10)" >> /workspace/koz3/orch/COORD_LOG.md
KOZ3_BOT9_LOCAL_TEST=1 KOZ3_BOT9_INPUT=/workspace/koz3/cpuk/ds KOZ3_BOT9_CODE=/workspace/koz3/cpuk/ds KOZ3_BOT9_WORK=/workspace/koz3/cpuk/box_s41/work \
 W3_SEEDS=41,42 KTHREADS=6 KOZ3_ALLOW_CPU_FALLBACK=1 HF_HUB_OFFLINE=1 \
 nice -n 10 /workspace/koz3/venv/bin/python ../k_usertest1234/koz3-w3-kurgan-cpu-seeds.py > run.log 2>&1
echo "- $(date '+%d.%m %H:%M MSK') box-cpu: kurgan seeds 41,42 exit=$?" >> /workspace/koz3/orch/COORD_LOG.md
