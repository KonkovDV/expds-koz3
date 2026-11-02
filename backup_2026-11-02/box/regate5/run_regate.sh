#!/bin/bash
# independent regate of GEO-ORG3 (code 823bda8 exported, own cache; bot5 caches symlinked read-only)
cd /workspace/koz3/regate5; PY=/workspace/koz3/w4venv/bin/python; export OMP_NUM_THREADS=4
export EXPDS_METRIC=/workspace/koz3/ds/e2e/score/compute_metrics.py
T=/workspace/koz3/w4geo/train; O=/workspace/koz3/regate5/oof; E=/workspace/koz3/regate5/eval; mkdir -p $E
[ -f $O/classic/200_Судьбищи_2.pkl ] || prlimit --as=7500000000 nice -n 19 $PY scripts/pix_clean_oof.py classic --root $T --config configs/v3_on.toml --cache $O --regions 200_Судьбищи_2 || echo FAIL classic200
for s in 1 2; do
  nice -n 19 $PY scripts/pix_org_oof.py fold --root $T --cache $O --seed $s --threads 4 || echo FAIL org$s
  nice -n 19 $PY scripts/pix_org_et_oof.py fold --root $T --cache $O --seed $s --threads 4 || echo FAIL orget$s
done
C="nice -n 19 $PY scripts/pix_clean_oof.py compose --root $T --config configs/v3_on.toml --cache $O"
P="--set geophys.pixgbm.enabled=true --set geophys.pixgbm.impl=\"clean\" --set geophys.pixgbm.maps=[\"etgr\",\"terfl\"] --tag s0"
$C --out $E/A18.geojson || echo FAIL A
for s in 0 1 2; do
  $C $P --orget-tag s$s --org-tag s$s --set 'geophys.pixgbm.org_classes=["finds_points"]' --out $E/Cetpts_orget_s$s.geojson || echo FAIL c$s
done
echo COMPOSED
