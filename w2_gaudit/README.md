# w2_gaudit — аудит «попадают ли кроны в DEM» (CI2-3 / P-W2-8, закрыт: нет)
- `k/` — ядро aaaa0000004/koz3-w2-ground-clip-audit (CPU), вход — датасет aaaa0000004/koz3-w2-gaudit-las.
- `out/gaudit/{015,021,186,190}.json`, `summary.json` — доли классов, какие точки идут в DEM; `out/*.log` — лог.
- `t/` — локальный тест на 018; `dl.log`, `up.log` — загрузка LAS/датасета.
