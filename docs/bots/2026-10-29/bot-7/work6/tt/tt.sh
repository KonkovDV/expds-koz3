#!/bin/bash
# bot-7 v3: защита времени аудита. (1) 18 участков (оптика SpOr/Ae + лидар + мелкие) — новый zip против 9138bff
# (старый порядок оптики), побайтно; (2) урезанный бюджет — новый zip с budget_s 200 / soft 150 / optics_s_per_gb 400
# и 9138bff с тем же бюджетом: файл ответа есть, валиден, что срезано. 1 процесс за раз, taskset 2 ядра, nice, RSS-сторож.
#   bash tt.sh <новый zip> <старый zip>   (старый = 8857343, до защиты времени; 9138bff удалён)
W=/workspace/koz3_shared/bot-7/work6; PY=/workspace/koz3_shared/bot-8/venv/bin/python; NEW=$1; OLD=$2; T=$(mktemp -d /tmp/b7tt_XXXX)
mkdir -p $T/new $T/old; (cd $T/new && unzip -q $NEW); (cd $T/old && unzip -q $OLD)
run() { # root tag
  rm -rf $T/o_$2; mkdir -p $T/o_$2
  ( cd $1 && taskset -c 0,1 $W/../work5/run.sh $W/tt/run_$2.log $PY inference/main.py $W/in/multi $T/o_$2/result.geojson )
  cp $T/o_$2/result.geojson $W/tt/res_$2.geojson 2>/dev/null
  echo "$2: $(tail -1 $W/tt/run_$2.log) | $(grep -o '"seconds": [0-9.]*}*$' $W/tt/run_$2.log | tail -1)"
}
# FR1h (frame_prior) в новом архиве ВКЛ.; в старом (8857343) его нет — для побайтной сверки защиты времени снимаем блок в копии
cp $T/new/configs/default.toml $T/new_default.toml
sed -i '/^\[post.frame_prior\]/,/^edge = /d' $T/new/configs/default.toml; grep -c "frame_prior\]" $T/new/configs/default.toml || true
run $T/new new_full; run $T/old old_full
cp $T/new_default.toml $T/new/configs/default.toml   # дальше — конфиг сдачи как есть (FR1h вкл.)
cmp $W/tt/res_new_full.geojson $W/tt/res_old_full.geojson && echo "MULTI BYTE-IDENTICAL" || echo "MULTI DIFF"
for r in new old; do
  sed -i 's/^budget_s = 1500 /budget_s = 200 /; s/^soft_ceiling_s = 1200/soft_ceiling_s = 150/; s/^optics_s_per_gb = 10.0/optics_s_per_gb = 400.0/' $T/$r/configs/default.toml
  grep -n "^budget_s\|^soft_ceiling_s\|^optics_" $T/$r/configs/default.toml
done
run $T/new new_cut; run $T/old old_cut
$PY $W/tt/tt_cmp.py $W/tt | tee $W/tt/tt_summary.txt
chmod -R u+w $T; rm -rf $T
