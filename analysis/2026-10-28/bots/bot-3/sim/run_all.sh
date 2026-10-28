#!/bin/bash
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
cd /workspace/koz3_shared/bot-3/sim
for sn in "0.5 0.15" "0.25 0.0" "0.25 0.15" "1.0 0.0" "1.0 0.15"; do set -- $sn
  ulimit -v 2000000; nice -n 10 ../venv/bin/python -c "import sim_poly as s; s.main(sigmas=($1,),noises=($2,))"
done
echo ALLDONE
