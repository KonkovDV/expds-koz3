# Help D: pull wsg2 when COMPLETE (B standby)
Kernel: dmitrijkv/koz3-w3-wsg2-cpu (lease bot-W3-kurgan → acc2/cpu-2).
Target: orch/bot-W3/wsg2_out/ (D owns; B only assists if D asks / A said help).
```bash
# status (acc2 = dmitrijkv)
orch/pool_kaggle.sh dmitrijkv kernels status dmitrijkv/koz3-w3-wsg2-cpu
# when COMPLETE:
mkdir -p orch/bot-W3/wsg2_out
orch/pool_kaggle.sh dmitrijkv kernels output dmitrijkv/koz3-w3-wsg2-cpu -p orch/bot-W3/wsg2_out
# then ping D+A; D releases slot/kernel
```
B does not touch GPU/q1/q2. Do not cancel/release D's lease.
