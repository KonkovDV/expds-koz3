#!/bin/bash
M=/workspace/koz3_shared/bot-9/kaggle/dfine_spor/full_rest_out/metric; B10=/workspace/koz3_shared/bot-10
export EXPDS_METRIC=$B10/repo/materials/compute_metrics.py OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 EXPDS_MATCH_SCOPE=global
PY=/workspace/rerank/venv/bin/python; ulimit -v 3000000
cd $M || exit 1; nice -n 10 $PY assemble.py > assemble.log 2>&1; echo "assemble rc=$?"
cd $B10 || exit 1
P() { nice -n 10 $PY work4/paired.py --pols drop,keep "$@"; }
P --base A2=$M/c2_A --b C=$M/c2_C --b E=$M/c2_E --b R=$M/c2_R --b R0=$M/c2_R0 --out $M/paired_2f.json > $M/paired_2f.log 2>&1; echo "p 2f rc=$? $(date +%T)"
P --base A=$M/c5_A --b C=$M/c5_C --b E=$M/c5_E --b R=$M/c5_R --b R0=$M/c5_R0 --out $M/paired_5f.json > $M/paired_5f.log 2>&1; echo "p 5f rc=$? $(date +%T)"
P --base C=$M/c5_C --b R=$M/c5_R --out $M/paired_5f_RvsC.json > $M/paired_5f_RvsC.log 2>&1; echo "p RvsC rc=$? $(date +%T)"
P --base A3=$M/c3_A --b C=$M/c3_C --b E=$M/c3_E --out $M/paired_3f_check.json > $M/paired_3f_check.log 2>&1; echo "p 3f rc=$? $(date +%T)"
cd $M && nice -n 10 $PY sp_compare5.py > sp_compare5.log 2>&1; echo "sp rc=$? $(date +%T)"
echo done
