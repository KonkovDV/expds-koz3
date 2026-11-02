#!/bin/bash
cd /workspace/koz3/wt/w1cfd
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 W5_DATA=/workspace/koz3/w1/cf/w5d PYTHONPATH=/workspace/koz3/w1/cf/ds/laspy-2.7.0-py3-none-any.whl
PY=/workspace/koz3/venv/bin/python; O=/workspace/koz3/w1/cf/inferpar; L=/workspace/koz3/w2/las/016_Хога; G=/workspace/koz3/w2/gaudit/ds
R=(--region "016_Хога:$L/01_Хога_La_точки/01_Хога_La_точки.las:$L/UTM.json" --region "186_Семибратнее_1:$G/186.las:$G/186_UTM.json"
   --region "190_Муравлево:$G/190.las:$G/190_UTM.json" --region 158_Курганы_29 --region 004_Боровичи_1)
while [ ! -f /workspace/koz3/w1/cf/w5d/work/nested_w5cfnc.npz ]; do sleep 20; done
for v in nocnx full; do
  oof=/workspace/koz3/w1/cf/w5d/work/nested_w5cf.npz; [ $v = nocnx ] && oof=/workspace/koz3/w1/cf/w5d/work/nested_w5cfnc.npz
  echo "== $v $(date +%H:%M:%S)"
  nice -n 10 $PY tools/w5cf_infer_parity.py --infer-dir /workspace/koz3/w1/cf/w5cf_infer_$v "${R[@]}" --oof $oof --json-out $O/p_$v.json > $O/log_$v.txt 2> $O/err_$v.txt
  echo "exit $?"
done

G=/workspace/koz3/w2/gaudit/ds
nice -n 10 $PY tools/w5cf_infer_bench.py --geojson /workspace/koz3/w1/cf/w5d/stack_arch.geojson --region 190_Муравлево --las $G/190.las --utm $G/190_UTM.json --infer-dir /workspace/koz3/w1/cf/w5cf_infer_nocnx > $O/bench_190.json 2> $O/bench_190.err; echo "bench exit $?"
echo ALLDONE
