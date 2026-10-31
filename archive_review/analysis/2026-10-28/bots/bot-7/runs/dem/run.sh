#!/bin/bash
# bot-7: продолжение грубого прохода городищ по ЦМР (копия bot-4/work/dem/run.sh).
# Код и venv bot-4 — только чтение; пишем в bot-7/work/dem. Пропускает готовые у bot-4 и у себя.
# По одному файлу, 2 ядра (taskset 0,1), nice 10, все пулы потоков = 2 (rayon 1), ulimit -v 3 ГБ.
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 NUMBA_NUM_THREADS=2 RAYON_NUM_THREADS=1 POLARS_MAX_THREADS=2
B=/workspace/koz3_shared/bot-4
D=/workspace/koz3_shared/bot-7/work/dem
cd $D
for p in $(find /workspace/data/train-expds-3 -name "*.la[sz]" -printf "%s %p\n" | sort -n | cut -d' ' -f2-); do
  reg=$(echo "$p" | cut -d/ -f5)
  [ -f $B/work/dem/out/$reg.geojson ] && continue
  [ -f out/$reg.geojson ] && continue
  # шлюз по памяти: ждать, пока MemAvailable >= 2,5 ГБ (машина общая)
  w=0; while [ $(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo) -lt 2500 ]; do
    [ $w -eq 0 ] && echo "$(date +%T) $reg жду память: $(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo) МБ"; w=$((w+10)); sleep 10; done
  [ $w -gt 0 ] && echo "$(date +%T) $reg ждал ${w}s"
  mkdir -p roots/$reg && ln -sfn /workspace/data/train-expds-3/$reg roots/$reg/$reg
  s=$(date +%s)
  ( ulimit -v 3000000; nice -n 10 taskset -c 0,1 $B/venv/bin/python $B/repo/scripts/hillfort_oof.py --root roots/$reg --out out/$reg.tmp --preset coarse ) > logs_$reg.txt 2>&1
  rc=$?
  [ $rc -eq 0 ] && mv out/$reg.tmp out/$reg.geojson
  echo "$(date +%T) $reg rc=$rc $(( $(date +%s)-s ))s $(tail -1 logs_$reg.txt | cut -c1-120)"
done
echo ALLDONE
