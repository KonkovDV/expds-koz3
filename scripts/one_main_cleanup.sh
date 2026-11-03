#!/usr/bin/env bash
# Свести репозиторий к одной ветке main (03.11.2026). Запускать с правами записи в KonkovDV/expds-koz3
# (облачная сессия Claude может пушить только в свою ветку — теги и удаление веток ей недоступны, HTTP 403).
#
#   bash scripts/one_main_cleanup.sh          # сухой прогон: что будет сделано
#   bash scripts/one_main_cleanup.sh --apply  # выполнить
#
# Для каждой удалённой ветки, кроме main и рабочей ветки облачной сессии (через неё изменения идут в main PR-ами):
#   1) проверить, что её конец входит в origin/main (иначе пропустить и сообщить — такую ветку сначала влить);
#   2) поставить тег archive/20261103/<ветка> на её конец и запушить (коммиты остаются адресуемыми);
#   3) удалить ветку на origin.
# bots/orch-protocol не удаляется, а перематывается на main (с ней работают боты Grok; при pull они получат main).
set -euo pipefail
APPLY=0; [ "${1:-}" = "--apply" ] && APPLY=1
KEEP_FF="bots/orch-protocol"
git fetch --prune origin '+refs/heads/*:refs/remotes/origin/*'
skipped=0; done_=0
for ref in $(git for-each-ref --format='%(refname:lstrip=3)' refs/remotes/origin | grep -v -x -e HEAD -e main -e claude/wizardly-brahmagupta-6mcd0s); do
  if ! git merge-base --is-ancestor "origin/$ref" origin/main; then
    echo "SKIP  $ref: есть коммиты вне main ($(git rev-list --count origin/main.."origin/$ref")) — сначала влить"; skipped=$((skipped+1)); continue
  fi
  tag="archive/20261103/$ref"
  if [ "$ref" = "$KEEP_FF" ]; then
    echo "FF    $ref -> main (тег $tag)"
    if [ $APPLY = 1 ]; then
      git tag -f "$tag" "origin/$ref" >/dev/null && git push -q origin "refs/tags/$tag"
      git push -q origin origin/main:"refs/heads/$ref"
    fi
    continue
  fi
  echo "DEL   $ref (тег $tag)"
  if [ $APPLY = 1 ]; then
    git tag -f "$tag" "origin/$ref" >/dev/null
    git push -q origin "refs/tags/$tag"
    git push -q origin --delete "$ref"
  fi
  done_=$((done_+1))
done
echo "веток к удалению: $done_, пропущено: $skipped$([ $APPLY = 0 ] && echo ' (сухой прогон: --apply, чтобы выполнить)')"
