#!/bin/bash
# W3: если текущий (v4/v5, без фикса) kurgan-cpu-seeds на аккаунте → ERROR, выпускаем наш v5-фикс (enqueue set ready). Только read-only status.
cd /workspace/koz3/orch
left="${W3_FAILOVER_ACCS:-usertest1234 aaaa0000004 aaaadg4gg4g4g}"
while [ -n "$left" ]; do
  nl=""
  for a in $left; do
    s=$(timeout 60 ./pool_kaggle.sh $a kernels status $a/koz3-w3-kurgan-cpu-seeds 2>&1 | tail -1)
    case "$s" in
      *ERROR*) python3 enqueue.py set w3-seeds-v5-$a status=ready >/dev/null && echo "$(date '+%d.%m %H:%M') $a ERROR → v5 ready";;
      *COMPLETE*) echo "$(date '+%d.%m %H:%M') $a COMPLETE (без фикса) — v5 не нужен";;
      *) nl="$nl $a";;
    esac
  done
  left="$nl"; [ -n "$left" ] && sleep 120
done
