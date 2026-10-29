#!/bin/bash
# CPU dry-run обеих голов (smoke-ядро) + проверка OOF-карт. Пик RSS — rss_watch.sh.
cd /workspace/koz3_shared/bot-9/kaggle/convnext_lidar
nice -n 10 python3 build.py > logs/build.log 2>&1 || { echo build_fail; exit 1; }
for H in centerhead heat; do
  o=$([ $H = heat ] && echo out_heat || echo out_ch); rm -rf _scratch/dry/$o
  CNX_HEAD=$H CNX_OUT=$PWD/_scratch/dry/$o ULIM=8000000 bash _scratch/dry_run.sh > logs/dry_run_$H.log 2>&1 & p=$!
  bash _scratch/rss_watch.sh $p logs/dry_run_${H}_rss.txt & wait $p; echo "exit=$?" >> logs/dry_run_$H.log
done
for d in out_ch out_heat; do echo "== $d"; OMP_NUM_THREADS=2 nice -n 10 _scratch/venv/bin/python tests/check_maps.py _scratch/dry/$d; done > logs/check_maps.log 2>&1
echo done
