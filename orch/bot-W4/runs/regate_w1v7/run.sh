#!/bin/bash
G=/workspace/koz3/orch/bot-W1/work/d3_gate.py; PY=/workspace/koz3/w1/venv/bin/python
V7='V7={"parts":[{"oof":"/workspace/koz3/out/w1-dfine-ae-final/oof_dfine_ae_flipwbf.geojson","markup":"ae","tta":"flipwbf","tau":0.7,"tau_cls":{"gorodishcha":1.01,"arkhitektury":1.01}},{"oof":"/workspace/koz3/w1/out/koz3-w1-dfine-or-final/oof_dfine_or_noflip.geojson","markup":"or","tta":"noflip","tau":0.6,"tau_cls":{"gorodishcha":1.01}}]}'
cd /workspace/koz3/orch/bot-W4/runs/regate_w1v7
nice -n 19 $PY $G --variant "$V7" --engine table --boot 1000 --sanity --ae-oof /workspace/koz3/out/w1-dfine-ae-final/oof_dfine_ae_noflip.geojson --out table_b1000.json > table_b1000.log 2>&1
nice -n 19 $PY $G --variant "$V7" --engine clusters --boot 1000 --out clusters_b1000.json > clusters_b1000.log 2>&1
nice -n 19 $PY $G --variant "$V7" --engine table --boot 10000 --also-joint --out table_b10000.json > table_b10000.log 2>&1
echo DONE > DONE
