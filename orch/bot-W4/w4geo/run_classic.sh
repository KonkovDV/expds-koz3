#!/bin/bash
# классика в кэш, по процессу на участок (пик памяти — одного участка); $@ — участки
cd /workspace/koz3/w4clean
for r in "$@"; do
  OMP_NUM_THREADS=3 nice -n 10 prlimit --as=${AS_LIMIT:-7500000000} /workspace/koz3/w4venv/bin/python scripts/pix_clean_oof.py classic \
    --root /workspace/koz3/w4geo/train --config configs/v3_on.toml --cache /workspace/koz3/w4geo/oof --regions "$r" || echo "FAIL $r"
done
echo CLASSICDONE
