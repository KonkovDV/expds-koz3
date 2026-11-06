#!/bin/bash
# YOLO queue (plan OPTICS_DETECTOR_PLAN): dataset ready → wait for c12-c push → free GPU slot → smoke (f4, 3 ep) →
# full default recipe (5 folds, 60 ep) → full gigaflex recipe after the ensemble kernel is done. Tokens never printed.
set -u
REPO=/home/user/expds-koz3; KG=$HOME/.local/bin/kaggle; B=/workspace/yolo_build; MK=$REPO/kaggle/2026-11-06/opt_yolo/make_yolo_kernel.py
st() { $KG kernels status dmitrijkv/$1 2>&1 | tail -1 | sed 's/.*status "//; s/"$//'; }
busy() { n=0; for k in koz3-li-c12-in-a koz3-li-c12-in-b koz3-li-c12-in-c koz3-li-ens-oof koz3-spor-yolo-smoke koz3-spor-yolo-def koz3-spor-yolo-gf; do
  s=$(st $k); echo "$s" | grep -qE "RUNNING|QUEUED" && n=$((n+1)); done; echo $n; }
pushretry() { while :; do o=$($KG kernels push -p "$1" 2>&1 | tail -1); echo "$(date -u +%H:%M) push $(basename $1): $o"; echo "$o" | grep -q "successfully pushed" && break; echo "$o" | grep -qi "quota" || break; sleep 1800; done; }
pushnew() { s=$(st $1); if echo "$s" | grep -qE "RUNNING|QUEUED|COMPLETE|ERROR"; then echo "$(date -u +%H:%M) $1 уже есть: $s"; else pushretry $B/$1; fi; }
waitslot() { while [ "$(busy)" -ge 2 ]; do sleep 180; done; }
waitdone() { while :; do s=$(st $1); echo "$s" | grep -qE "COMPLETE|ERROR|CANCEL" && break; sleep 120; done; echo "$s"; }
echo "$(date -u +%H:%M) start"
for i in $(seq 1 60); do s=$($KG datasets status dmitrijkv/koz3-spor-yolo 2>&1 | tail -1); echo "$s" | grep -q ready && break; sleep 60; done
echo "$(date -u +%H:%M) dataset: $s"
while ! grep -qE "push koz3-li-c12-in-c: .*successfully|c уже есть|СТОП" /workspace/autopilot_c12.log; do sleep 120; done
waitslot
python3 $MK --slug koz3-spor-yolo-smoke --folds 4 --epochs 3 --out $B >/dev/null && pushnew koz3-spor-yolo-smoke
sleep 60; s=$(waitdone koz3-spor-yolo-smoke); echo "$(date -u +%H:%M) smoke: $s"
mkdir -p /workspace/yolo_out/smoke && (cd /workspace/yolo_out/smoke && $KG kernels output dmitrijkv/koz3-spor-yolo-smoke -p . >/dev/null 2>&1)
python3 -c "
import json,sys
t=open('/workspace/yolo_out/smoke/koz3-spor-yolo-smoke.log',encoding='utf-8').read()
try: s=''.join(x.get('data','') for x in json.loads(t))
except Exception: s=t
print(s[-1500:]); sys.exit(0 if 'ИТОГ: OK' in s else 1)" > /workspace/yolo_out/smoke/tail.txt 2>&1 \
  || { echo "$(date -u +%H:%M) СТОП: smoke YOLO не OK"; tail -5 /workspace/yolo_out/smoke/tail.txt; exit 1; }
echo "$(date -u +%H:%M) smoke OK: $(grep -E 'фолд|окон' /workspace/yolo_out/smoke/tail.txt | tail -2 | tr '\n' ' ')"
waitslot
python3 $MK --slug koz3-spor-yolo-def --folds 0,1,2,3,4 --epochs 60 --recipe default --out $B >/dev/null && pushnew koz3-spor-yolo-def
while ! grep -qE "^[0-9:]+ end|СТОП" /workspace/autopilot_ens.log; do sleep 180; done
waitslot
python3 $MK --slug koz3-spor-yolo-gf --folds 0,1,2,3,4 --epochs 60 --recipe gigaflex --out $B >/dev/null && pushnew koz3-spor-yolo-gf
for k in koz3-spor-yolo-def koz3-spor-yolo-gf; do s=$(waitdone $k); echo "$(date -u +%H:%M) $k: $s"
  mkdir -p /workspace/yolo_out/$k && (cd /workspace/yolo_out/$k && $KG kernels output dmitrijkv/$k -p . >/dev/null 2>&1); done
echo "$(date -u +%H:%M) end"
