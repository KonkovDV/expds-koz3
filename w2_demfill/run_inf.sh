#!/bin/bash
# P-W2-10: CPU-инференс c3-sc весов своих фолдов, 1 процесс × 3 потока (nproc=3, бокс загружен)
cd /workspace/koz3/w2/demfill
declare -A F=([015_Демидовка]=2 [021_Новенькое]=3 [186_Семибратнее_1]=3 [190_Муравлево]=4)
wpath(){ s=$1; f=$2; case $s in
 s0) [ $f = 4 ] && echo w/s0/koz3-cnx-li-c3sc-f4-v2/weights_cnx_c3-sc_f4.pt || echo w/s0/koz3-cnx-li-c3sc-f23/weights_cnx_c3-sc_f$f.pt;;
 s1) [ $f = 4 ] && echo w/s1/koz3-cnx-li-c3sc-s1f4/weights_cnx_c3-sc_f4.pt || echo w/s1/koz3-cnx-li-c3sc-s1f23/weights_cnx_c3-sc_f$f.pt;;
 s2) [ $f = 4 ] && echo ../oof/s2/koz3-w2-cnx-c3sc-s2f4/weights_cnx_c3-sc_f4.pt || echo ../oof/s2/koz3-w2-cnx-c3sc-s2f23/weights_cnx_c3-sc_f$f.pt;; esac; }
for site in 186_Семибратнее_1 015_Демидовка 021_Новенькое 190_Муравлево; do f=${F[$site]}
 for arm in T C; do for sd in s0 s1 s2; do o=maps/$arm/$sd/maps_cnx_$site.npz; [ -f $o ] && continue; mkdir -p maps/$arm/$sd
  echo "== $(date +%T) $arm $sd $site f$f"
  OMP_NUM_THREADS=3 nice -n 10 /workspace/koz3/venv/bin/python infer_cpu.py --relief rel/$arm/$site.npy --meta rel/$arm/$site.json --weights $(wpath $sd $f) --fold $f --out $o --threads 3 2>&1 | grep -E "^wrote|Error|error|/[0-9]+ tiles" | tail -2
 done; done; done
echo ALLDONE $(date +%T)
