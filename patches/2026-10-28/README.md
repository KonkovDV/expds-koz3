# Патчи ботов 28.10.2026 (не применены к main)

Это предложения, их принимают по `docs/EXPERIMENTS.md`. Ниже результат `git apply --check` на 24bd916.

- Применяются чисто: bot-3 (`audit01`, `geophys`), bot-4 `gorfort`, bot-5 `arch_microcircle` (отклонено), bot-6 `optics`, `twins-only` (ждёт проверки правил), bot-7 (`bundle`, `bundle_measure_tools`, `candidate`, `geom_opt`), bot-8 (`reliability`, `rust-metric`, `for-bot7*`), bot-9, bot-10 `honest_bootstrap`.
- Ставятся только поверх других патчей: bot-10 `site_scope_provisional` (`metric.py`), `unseen_keep_tooling` (нужен `honest_bootstrap`), bot-8 `wip_speed_cache_2040` (нужен `rust-metric`), `site-scope-worktree` (бинарная правка `.so` без индекса — собрать из исходников).
