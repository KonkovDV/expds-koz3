#!/bin/bash
# one process per region (lazrs crashed with free() when reusing a process)
cd /workspace/koz3_shared/bot-2/repo
ulimit -v 2500000
export OMP_NUM_THREADS=1
for d in /workspace/data/train-expds-3/*/01_*/; do
  r=$(basename $(dirname $d))
  ls $d/*.la? >/dev/null 2>&1 || continue
  [ -f ../adaf/slrm/$r.npz ] && continue
  nice -n 10 ../venv/bin/python -X faulthandler scripts/adaf/prep_slrm.py --out ../adaf/slrm --only $r || echo "FAIL $r $?"
done
echo done
