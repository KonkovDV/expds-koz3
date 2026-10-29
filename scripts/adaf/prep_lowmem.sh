#!/bin/bash
# after prep_all.sh: redo regions that FAILed or got valid_frac 0 (no class 2). RAM/Qhull failures -> --lowmem.
cd /workspace/koz3_shared/bot-2/repo
while ! grep -q '^done' ../adaf/prep_slrm.log; do sleep 20; done
ulimit -v 2500000
export OMP_NUM_THREADS=1
LOW="037_Никола_Ленивец 069_Стеклянуха_2 105_Вщиж"
for r in $(grep '^FAIL' ../adaf/prep_slrm.log | awk '{print $2}') 101_Гнёздово_Днепровская 102_Гнёздово_Заольшанская; do
  [ -f ../adaf/slrm/$r.npz ] && continue
  fl=""; case " $LOW " in *" $r "*) fl="--lowmem";; esac
  nice -n 10 ../venv/bin/python -X faulthandler scripts/adaf/prep_slrm.py --out ../adaf/slrm --only $r $fl || echo "FAIL $r $?"
done
echo done
