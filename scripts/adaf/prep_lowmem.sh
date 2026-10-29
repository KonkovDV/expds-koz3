#!/bin/bash
# after prep_all.sh: retry regions that FAILed (RAM / Qhull abort) with --lowmem, one process each
cd /workspace/koz3_shared/bot-2/repo
while ! grep -q '^done' ../adaf/prep_slrm.log; do sleep 20; done
ulimit -v 2500000
export OMP_NUM_THREADS=1
for r in $(grep '^FAIL' ../adaf/prep_slrm.log | awk '{print $2}'); do
  [ -f ../adaf/slrm/$r.npz ] && continue
  /usr/bin/time -f "maxrss_kb %M $r" nice -n 10 ../venv/bin/python -X faulthandler scripts/adaf/prep_slrm.py --out ../adaf/slrm --only $r --lowmem || echo "FAIL $r $?"
done
echo done
