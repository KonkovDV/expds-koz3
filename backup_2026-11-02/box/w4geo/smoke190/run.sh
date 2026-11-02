#!/bin/bash
X=/workspace/koz3/w4geo/x_v4on; S=/workspace/koz3/w4geo/smoke190; O=/workspace/koz3/w4geo/v4par
mkdir -p $O; cd $X
export OMP_NUM_THREADS=4 W5_DATA=/workspace/koz3/w1/cf/w5d PYTHONPATH=/workspace/koz3/w1/cf/ds/laspy-2.7.0-py3-none-any.whl
PY=/workspace/koz3/venv/bin/python; L=/workspace/koz3/w2/las/016_Хога; G=/workspace/koz3/w2/gaudit/ds
echo "== parity $(date +%H:%M:%S)"
$PY -c "import laspy;print('laspy',laspy.__version__,laspy.__file__)"
$PY tools/w5cf_infer_parity.py --infer-dir $X/models/w5cf_infer \
 --region "016_Хога:$L/01_Хога_La_точки/01_Хога_La_точки.las:$L/UTM.json" --region "186_Семибратнее_1:$G/186.las:$G/186_UTM.json" \
 --region "190_Муравлево:$G/190.las:$G/190_UTM.json" --region 158_Курганы_29 --region 004_Боровичи_1 \
 --oof /workspace/koz3/w1/cf/w5d/work/nested_w5cfnc.npz --json-out $O/p_archive.json > $O/log_par.txt 2> $O/err_par.txt
echo "parity exit $?"
echo "== smoke $(date +%H:%M:%S)"
unset PYTHONPATH W5_DATA OMP_NUM_THREADS
export PYTHONPATH=/workspace/koz3/w1/cf/ds/laspy-2.7.0-py3-none-any.whl
/usr/bin/time -v $PY inference/main.py $S/in $S/out_on/result.geojson > $S/smoke_on.log 2> $S/smoke_on.err
echo "smoke exit $?"
echo ALLDONE $(date +%H:%M:%S)
