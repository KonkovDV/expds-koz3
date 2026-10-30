# Архив исследований и результатов KOZ3 (29.10.2026, ~20:45 МСК)

Полная копия рабочего каталога оркестрации (shared/koz3) на момент остановки.

- Лучший LB: 0.41238 (submission_bot7cand_v2_eb268d6.zip, 29.10 08:47), #25 из 72; лидер 0.61636.
- Разбивка LB: reports/lb/scoring_v2_eb268d6_0.41238.log. Диагноз aerial/ortho: reports/ae_ortho_diagnosis.md.
- BOARD.md: статус и фокус; BACKLOG.md: задачи по ботам; PROTOCOL.md: правила; LESSONS.md: выученные уроки.
- EXPERIMENTS.csv: журнал экспериментов; reports/: отчёты ботов; pc/, box/, kaggle/: очереди, раннеры, результаты.
- Ветки ботов с кодом: BOT_BRANCHES.txt (draft PR, main не трогали).

Незавершённое на момент остановки:
- bot3 A+B (D-FINE Ae, f4pack) @1e7073a: готов, ждал Docker-гейт на ПК (ожидаемо +0.02–0.05);
- bot2 sat_thr_relax @491c63e: готов к гейту (+0.003–0.010);
- bot8 TTA @f6fe20a (PR #9): готов к гейту;
- bot4: недостающий вариант абляции on_all_except_coarse, area-aware τ городищ;
- bot6: ортофото multi-class (OOF dfine-or);
- Kaggle acc2: c3sc-s1f23 / s1f4, e2e-калибровка koz3-e2e-eval и bot8 OOF honest-10 не запускались.
