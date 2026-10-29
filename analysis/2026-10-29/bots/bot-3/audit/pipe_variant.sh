#!/bin/bash
# bash pipe_variant.sh NAME [ENV=VAL ...]: extract → train → predict (LOFO) в audit/var/NAME, тем же кодом (worktree bot3-audit)
set -u
N=$1; shift
for kv in "$@"; do export "$kv"; done
A=/workspace/koz3_shared/bot-3/audit; V=$A/var/$N; mkdir -p $V
D=/workspace/data/train-expds-3
env | grep PIXGBM_ > $V/env.txt
$A/run_a.sh ${N}_extract scripts/pixgbm.py extract --root $D --out $V/feats || exit 1
grep -q "exit=0" $A/out/${N}_extract.rss || exit 1
$A/run_a.sh ${N}_train scripts/pixgbm.py train --feats $V/feats --models $V/models
grep -q "exit=0" $A/out/${N}_train.rss || exit 1
$A/run_a.sh ${N}_predict scripts/pixgbm.py predict --root $D --models $V/models --out $V/pix_lofo.geojson
echo DONE $N $(date) >> $A/out/pipe.log
