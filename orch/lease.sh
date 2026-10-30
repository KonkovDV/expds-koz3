#!/bin/bash
# Аренды KOZ3 под эксклюзивным flock (атомарно на боксе).
#   orch/lease.sh acquire <res> <owner> <ttl_h> [--note "..."] [--kernel acc/slug]
#   orch/lease.sh renew   <res> <owner> <ttl_h>
#   orch/lease.sh release <res> <owner> [--force  (только coord)]
#   orch/lease.sh list [--all] | who <res>
#   orch/lease.sh sync    — зеркало LEASES.json/COORD_LOG.md в ветку bots/orch-protocol (worktree wt/coord-orch)
# res: slot:<acc>/<gpu-1|gpu-2|tpu|cpu> | kernel:<acc>/<slug> | dataset:<acc>/<slug> | branch:<name>
# Пример: orch/lease.sh acquire slot:acc1/gpu-1 bot-W1-aeor 8 --kernel elenakonkova/koz3-w1-dfine-ae-final --note "ae final"
set -euo pipefail
ORCH="${KOZ3_ORCH:-/workspace/koz3/orch}"
LOCK="$ORCH/.leases.lock"
WT="${KOZ3_ORCH_WT:-/workspace/koz3/wt/coord-orch}"
cmd="${1:-list}"
if [[ "$cmd" == sync ]]; then
  exec 9>"$LOCK"; flock -w 60 9
  cd "$WT"
  git fetch -q origin bots/orch-protocol && git rebase -q --autostash origin/bots/orch-protocol || { echo "rebase failed"; exit 1; }
  cp "$ORCH/LEASES.json" "$ORCH/COORD_LOG.md" orch/
  if grep -qE 'KGAT_[A-Za-z0-9]{8,}|ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}' orch/LEASES.json orch/COORD_LOG.md; then echo "SECRET PATTERN — отмена"; exit 2; fi
  git add orch/LEASES.json orch/COORD_LOG.md
  git -c user.name=coord-lease -c user.email=coord@koz3.local commit -qm "orch: leases snapshot $(date '+%d.%m %H:%M MSK')" || { echo "без изменений"; exit 0; }
  git push -q origin HEAD:bots/orch-protocol && echo "synced $(git rev-parse --short HEAD)"
  exit 0
fi
exec 9>"$LOCK"
flock -w 30 9 || { echo "lock timeout"; exit 1; }
python3 "$(dirname "$(readlink -f "$0")")/lease.py" "$@"
