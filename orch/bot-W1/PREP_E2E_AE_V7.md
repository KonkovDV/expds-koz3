# PREP e2e Ae V7 (бот D / W1) — 30.10 13:00 MSK, только CPU/подготовка

V7 = ae flipwbf τ0.7 (все классы), gor/arch ae выкл (1.01) + or noflip τ0.6, gor or выкл.
Гейт W1 (04:05): Δ+0.0033 P.935, пол P4 ок, new34 +0.0004, nolidar +0.0087 → PASS по одиночному гейту;
MASTER_PLAN/STATE:8: V7 REJECTED (Бонферрони/7). Цель e2e — честная проверка gate↔docker (знак совпадает?), НЕ принятие в архив.

## Есть на боксе
- Код гейта-результаты: wt/w1/analysis/2026-10-30/bots/bot-W1/d3/runs/{ae_final_prereg_wsa1,or_final_V6,...}.json/.log
- PREREG: wt/w1/docs/bots/2026-10-30/bot-W1/PREREG_W1.md (V7 стр.43, флаги ae_nolidar_only/ae_max_det/or_alpha_any_dtype)
- Метрика: wt/w1/materials/compute_metrics.py (config.py НЕТ — гейт идёт на встроенных дефолтах)
- Ядра: wt/w1/docs/bots/2026-10-30/bot-W1/kernels/koz3-w1-dfine-{ae,or}-final (+ -v2)
- or-final: только метаданные (dfine_f0..2, dfine_or = 12K, без весов), log, dfine_report.json — wt/w1/analysis/.../out/koz3-w1-dfine-or-final
- Kaggle (read-only status 13:00): elenakonkova/koz3-w1-dfine-ae-final COMPLETE, elenakonkova/koz3-w1-dfine-or-final COMPLETE → выход скачиваем
- Деплой-флаг: [dfine_aeor] enabled=false + [thresholds_by_markup.dfine_or] gorodishcha=1.01 — в configs/default.toml ветки origin/bots/bot5-candidate-v4 (50f6f27, линия aba432d). В bots/w1-aeor (1b457f3 от main 70dedf4) секции [dfine_aeor] НЕТ.
- docker/Dockerfile в wt/w1

## НЕТ (блокеры)
1. d3_gate.py — был orch/bot-W1/work/d3_gate.py, каталога work/ нет, в git не закоммичен (ни в одной ветке). Нужен у бота 1/W4 или пересобрать.
2. table.pkl (/workspace/koz3/w5data/work/table.pkl) — нет; w5data/work пуст; pc_data/ пуст (ПК offline; тот же блокер у W4 паритета v2).
3. gt_train_v2.geojson, stack_arch.geojson(.gz) в w5data — нет (sha256 эталона stack_arch 42ff3087…).
4. OOF: /workspace/koz3/out/w1-dfine-ae-final/oof_dfine_ae_{noflip,flipwbf}.geojson и /workspace/koz3/w1/out/koz3-w1-dfine-or-final/oof_dfine_or_noflip.geojson — нет (бокс пересоздан). Источник — kernels output (см. ниже).
5. Веса ae (dfine_final + f0–f4, 6×~125 МБ) и or (final+f0–f2) для docker — только в выходе ядер.
6. Порог/флаги V7 в конфиге: нужна ветка/патч поверх v4: [dfine_aeor] enabled=true, ae τ 0.7 все классы, ae gor/arch 1.01, ae TTA=flipwbf (WBF 0.55); or τ 0.6, TTA noflip, or gor 1.01. Точные ключи взять из expds (dfine_aeor, ветка bot5-candidate-v4).
7. Docker mock-e2e — владелец W4 (SCHEDULE: D-FINE вне архива до согласия gate↔docker); config.py метрики платформы.

## Команды после 03:05 31.10 (download-часть — CPU, можно и раньше с разрешения A)
```
cd /workspace/koz3
# 0) аренда — только через A: попросить бота 1 выдать (для e2e GPU не нужен; docker CPU)
orch/lease.sh list
# 1) выход ядер (read-only, CPU)
mkdir -p out/w1-dfine-ae-final w1/out/koz3-w1-dfine-or-final
orch/pool_kaggle.sh elenakonkova kernels output elenakonkova/koz3-w1-dfine-ae-final -p out/w1-dfine-ae-final
orch/pool_kaggle.sh elenakonkova kernels output elenakonkova/koz3-w1-dfine-or-final -p w1/out/koz3-w1-dfine-or-final
ls out/w1-dfine-ae-final/oof_dfine_ae_*.geojson w1/out/koz3-w1-dfine-or-final/oof_dfine_or_noflip.geojson
# 2) table.pkl / GT / stack_arch — от бота 1 / W4 / ПК → w5data/{work/table.pkl,gt_train_v2.geojson,stack_arch.geojson.gz}; sha256sum stack_arch == 42ff3087…
# 3) d3_gate.py → orch/bot-W1/work/; санити: A honest-10 == 0.3146153227, затем повтор V7 (сверка с ae_final_prereg_wsa1.json)
# 4) e2e: ветка от bots/bot5-candidate-v4 локально (НЕ push), патч конфига V7, models/ ← веса final; docker build/run на моке (W4), time ≤22 мин
# 5) сравнить знак Δ docker vs gate; отчёт A ≤10 строк. Если GPU понадобится — только orch/lease.sh через A.
```

## Update 30.10 13:03 MSK
- OOF+weights ae/or скачаны (пути выше).
- GT: w5data/koz3-gt-train-v2/gt_train_v2.geojson (+ symlink w5data/gt_train_v2.geojson).
- Остаётся hold: table.pkl, stack_arch, d3_gate.py (ПК Елены / A).
