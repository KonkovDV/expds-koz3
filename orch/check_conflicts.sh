#!/bin/bash
# Детектор конфликтов: двойные слоты/ядра, истёкшие аренды, ядра без аренды (--kaggle),
# общий клон не на main / грязный, worktree вне wt/<owner> или на чужой ветке, чужие коммиты в ветках (--git).
# Использование: orch/check_conflicts.sh [--git] [--kaggle] [--all] [--log]   (cron/watcher: --all --log каждые 15 мин)
exec nice -n 10 python3 "$(dirname "$(readlink -f "$0")")/check_conflicts.py" "$@"
