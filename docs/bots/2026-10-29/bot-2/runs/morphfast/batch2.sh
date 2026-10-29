set -e
P="../../venv/bin/python -X utf8 scripts/morph_fast.py --cands ../../../bot-4/work/cropverif/li_kurgan_candidates.npz"
export OMP_NUM_THREADS=1
nice -n 10 $P --out ../morphfast/full --only 028_Вышегжа_2
for c in 0.005 0.01 0.02; do
 for r in 001_Гнёздово_Нивленское 028_Вышегжа_2; do nice -n 10 $P --out ../morphfast/conf$c --conf-min $c --only $r; done
done
nice -n 10 $P --out ../morphfast/demli --dem-from-li --only 001_Гнёздово_Нивленское
echo ALLDONE
