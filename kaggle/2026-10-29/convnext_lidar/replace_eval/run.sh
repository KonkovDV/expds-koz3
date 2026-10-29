#!/bin/bash
# replace_eval — одна команда:  bash run.sh            (все OOF /workspace/kaggle_out/koz3-cnx-li-c3*/oof_cnx_*.geojson без *_fN)
#   VARIANTS="add rep_raw rep_mix rep_rank rep_quantile" CNX_OOF="a.geojson b.geojson" RE_WORK=dir bash run.sh
# Шаги: GT затронутых семей → fam_V (jsonl) → family_curves бота 10 → c_V (100 юнитов) → paired.py (A vs V, drop/keep) → report.py; jsonl удаляются.
set -u
R=$(cd "$(dirname "$0")" && pwd); B10=/workspace/koz3_shared/bot-10; export RE_WORK=${RE_WORK:-$R/work}; W=$RE_WORK; mkdir -p $W
VARIANTS=${VARIANTS:-"add rep_raw rep_mix rep_rank rep_quantile"}
export EXPDS_METRIC=$B10/repo/materials/compute_metrics.py OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 EXPDS_MATCH_SCOPE=global
PY=/workspace/rerank/venv/bin/python; CL=$B10/work/family_clusters.json
ulimit -v 3000000
fc() { (cd $B10 && nice -n 10 $PY repo/scripts/family_curves.py --clusters $CL "$@"); }
dfs() { echo "df $(df -h / | tail -1 | awk '{print $4}') $(date +%T) $1"; }
rmfam() { find $W/fam_$1 -type f -name '*.jsonl' -delete; find $W/fam_$1 -type l -delete; rmdir $W/fam_$1; dfs "rm fam_$1"; }
# стек бота 10 по семьям — своя копия (bot-10/work/fam удалён 29.10 08:21); удаляется в конце
export FAM_SRC=${FAM_SRC:-$W/fam_src}; OWN_SRC=0
if [ ! -s $FAM_SRC/_counts.json ]; then OWN_SRC=1; rm -rf $FAM_SRC; (cd $B10/repo && nice -n 10 $PY scripts/split_stack_by_family.py --stack /workspace/rerank/stack_arch.geojson --out $FAM_SRC) > $W/fam_src.log 2>&1 || { echo "split FAIL"; exit 1; }; dfs "fam_src $(tail -1 $W/fam_src.log)"; fi
if [ "${STAGE:-all}" = all ]; then
nice -n 10 $PY $R/prep.py gt > $W/gt.log 2>&1 || { echo "gt FAIL"; tail $W/gt.log; exit 1; }; dfs start
for V in base $VARIANTS; do
  rm -rf $W/curves_$V
  nice -n 10 $PY $R/prep.py fam $V > $W/fam_$V.log 2>&1 || { echo "fam $V FAIL"; tail $W/fam_$V.log; exit 1; }; dfs "fam $V"
  fc --fam $W/fam_$V --gt $W/gt_t.geojson --out $W/curves_$V > $W/curves_$V.log 2>&1 || { echo "curves $V FAIL"; tail $W/curves_$V.log; exit 1; }
  echo "curves $V ok $(date +%T)"; rmfam $V
done
nice -n 10 $PY $R/build.py $VARIANTS > $W/build.log 2>&1 || { echo "build FAIL"; tail $W/build.log; exit 1; }; cat $W/build.log
fi  # STAGE=paired: только paired + report по готовым c_V
# paired по одному варианту (5 сразу: 1,5 ГБ RSS, убит OOM 29.10 08:35)
for V in $VARIANTS; do
  (cd $B10 && nice -n 10 $PY work4/paired.py --pols drop,keep --base A=$W/c_A --b $V=$W/c_$V --out $W/paired_$V.json > $W/paired_$V.log 2>&1); echo "paired $V rc=$? $(date +%T)"
done
nice -n 10 $PY $R/report.py $VARIANTS > $W/report.log 2>&1; echo "report rc=$? $(date +%T)"; tail -3 $W/report.log
rm -rf $W/curves_base; [ "${OWN_SRC:-0}" = 1 ] && [ -z "${KEEP_SRC:-}" ] && rm -rf $FAM_SRC; dfs done
