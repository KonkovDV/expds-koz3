# Бюджет времени — release v5

Цель: wall ≤ **20 мин (1200 с)** на тестовом объёме (LB лимит 30 мин; база eb268d6 = **3:48 ≈ 228 с**).

`configs/default.toml` → `[time] soft_ceiling_s = 1200` (уже в базе). Сторож: `scripts/budget_guard.py`.

## Оценка стоимости фич (добавка к базе ~228 с)

| фича | статус | est. Δ wall | v5 |
|---|---|---:|---|
| PIXGBM + rerank (база) | ON / LB | 0 (в 228 с) | включено |
| satellite SpOr D-FINE | REJECTED / OFF | +60–180 с | **исключено** |
| aerial + ortho (dfine_aeor Ae) | unproven / OFF | +120–300 с | **исключено** |
| dfine_aeor Or-only | pending (c′) | +60–180 с | **OFF** до Δ>0 |
| large_objects | REJECTED Δ≈−0.016 | +10–40 с | **исключено** |
| TTA / MS (bot8) | unproven e2e | +30–90 с | **исключено** |
| gor_dense | REJECTED Δ=0 | ~0 (no-op) | **исключено** |
| validator / sha / manifest | tooling | <1 с | включено |

**Сумма v5 (tooling-only):** ≈ база **228 с ≪ 1200 с**. Запас ~16 мин на будущие ACCEPTED фичи.

## Guard

```bash
python scripts/budget_guard.py --config configs/default.toml
python scripts/budget_guard.py --wall-sec 228 --max-sec 1200
```

Exit 0 = ok; 1 = over budget / soft_ceiling > max.
