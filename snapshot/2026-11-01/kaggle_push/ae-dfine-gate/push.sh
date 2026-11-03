#!/usr/bin/env bash
# Push ae-dfine-gate kernel only if the account has a free GPU slot (max 2 ACTIVE).
# Usage: push.sh elenakonkova|dmitrijkv
set -euo pipefail
ACC="${1:?account}"
ROOT="$(cd "$(dirname "$0")" && pwd)"
SEC="$HOME/.secrets/kpool/$ACC"
if [[ ! -f "$SEC" ]]; then echo "no token $SEC"; exit 2; fi
export KAGGLE_API_TOKEN=$(cat "$SEC")
KAGGLE="${KAGGLE:-$HOME/.venvs/kaggle/bin/kaggle}"
running=0
while read -r ref; do
  [[ -z "$ref" ]] && continue
  st=$($KAGGLE kernels status "$ref" 2>/dev/null | tr -d '\r' || true)
  if echo "$st" | grep -qiE 'RUNNING|QUEUED|PENDING'; then
    running=$((running+1))
    echo "ACTIVE $ref :: $st"
  fi
done < <($KAGGLE kernels list --mine --page-size 40 2>/dev/null | awk 'NR>2 {print $1}' | head -35)
echo "active_count=$running account=$ACC"
if [[ "$running" -ge 2 ]]; then
  echo "BUSY: $running sessions — not pushing; re-run: $0 $ACC"
  exit 3
fi
DIR="$ROOT/$ACC"
if [[ ! -f "$DIR/kernel-metadata.json" ]]; then echo "missing $DIR"; exit 2; fi
echo "pushing $DIR ..."
$KAGGLE kernels push -p "$DIR"
echo "pushed OK $(date -Iseconds)"
$KAGGLE kernels status "$ACC/koz3-w1-ae-dfine-gate" || true
