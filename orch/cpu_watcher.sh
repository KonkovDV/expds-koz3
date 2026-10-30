#!/bin/bash
# waits for pixgbm-s3, pulls output, releases acc2/cpu lease
cd /workspace/koz3
while true; do
  s=$(orch/pool_kaggle.sh dmitrijkv kernels status dmitrijkv/koz3-w4-pixgbm-s3 2>&1 | tail -1)
  case "$s" in *COMPLETE*|*ERROR*|*CANCEL*) break;; esac
  sleep 600
done
mkdir -p w4pix/out2 && orch/pool_kaggle.sh dmitrijkv kernels output dmitrijkv/koz3-w4-pixgbm-s3 -p w4pix/out2 -q >/dev/null 2>&1
st=$(head -c 200 w4pix/out2/pixgbm_retrain/STATUS.txt 2>/dev/null)
python3 - "$s" "$st" <<'PY'
import json,sys,datetime
L=json.load(open('orch/LEASES.json'))
for x in L['slots']:
    if x['slot']=='acc2/cpu':
        x.update({'holder':None,'kernel':None,'state':'свободен','last':f"koz3-w4-pixgbm-s3 {sys.argv[1].split('.')[-1]} {sys.argv[2][:120]} → w4pix/out2 (NON-CLEANROOM)"})
json.dump(L,open('orch/LEASES.json','w'),ensure_ascii=False,indent=1)
with open('orch/COORD_LOG.md','a') as f: f.write(f"- {datetime.datetime.now():%d.%m %H:%M} MSK | cpu_watcher | pixgbm-s3 {sys.argv[1].split('.')[-1]}; STATUS: {sys.argv[2][:120]}; вывод w4pix/out2; аренда acc2/cpu снята\n")
PY
