# w2_demfill — P-W2-10 «заполнение дыр DEM» (флаг `lidar.dem_fill="tin_lowveg"`, по умолчанию "delaunay")
Порядок: `gapstats.py` (доля дыр по участкам → `out/gapstats.json`) → `dlw.sh` (скачать веса c3-sc) → `build_relief.py` (рельеф 015/021/186/190 с новым заполнением) → `run_inf.sh`/`infer_cpu.py` (CPU-инференс весами СВОЕГО фолда) → `swap_sites.py` (подменить эти участки в OOF) → `liscore.py` (Li-only оценка, `out/liscore_K.json`) → `run_gate.sh` (гейт). Логи `*.log`. Условия PASS — PREREG P-W2-10. Большие выходы — бокс `/workspace/koz3/w2/demfill/{rel,maps,out}` → Kaggle elenakonkova/koz3-w2-demfill (когда тест закончится).

## Итог: FAIL
T (tin_lowveg) против C (delaunay): Δ+0,0014, P(T>C)=0,99, ниже порога +0,003; полы не нарушены. Флаг остаётся выключенным. Выходы: Kaggle `elenakonkova/koz3-w2-demfill` (`rel/{C,T}`, `maps/{C,T}/s{0,1,2}`, `out/result_P-W2-10.json`, гейт, liscore; описание — `DATASETS.md`).

**Открытая побочная находка (проверить, низкий приоритет):** на 190_Муравлево карты s1 f4 из Kaggle не совпадают с инференсом по скачанным весам s1f4. Li F1 на 190: 0,666 (C) против 0,515 в текущем OOF (K). Возможно, ядро сохранило не те веса или OOF s1 на f4 занижен. См. HANDOFF, п. 10.
