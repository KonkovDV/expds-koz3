# Резервный вариант D-FINE final (НЕ СОБИРАТЬ без приёмки bot-10)

**Модель сдачи v3 — f0** (`/workspace/kaggle_out/koz3-dfine-spor-full-rest/dfine_f0`, model.safetensors sha256
`81ef8228…09ed8`). Решение bot-10 29.10: final — только резерв.

Final: `/workspace/kaggle_out/koz3-dfine-spor-final/dfine_final/` — model.safetensors sha256
`bdb579ca8928a2d99d179b8d52cbddf63a925274d73b2f5fcc2db6e943050729` (проверено bot-7 29.10),
config.json `ff8bb630…0789` (bot-9 FOR_BOT7_V3.md §7). Брать fp32-веса; fp16 только через autocast
(`expds/sp_dfine.py` так и делает; `.half()` в коде нет — тест `test_sp_dfine_fp32_weights_autocast_no_half`).
T4 (bot-9): fp16 autocast bs8 22,3 мс/окно (44,7 с с flip на 64 окна).

## Почему это не замена одной строкой
- Пороги `[thresholds_by_markup.sp_dfine]` подобраны по OOF фолдовых моделей (f0-семейство). У final OOF нет
  (обучен на всех SpOr), а распределение уверенностей у него другое (bot-9 §7: вторая детекция 0,625 против 0,825 у f0).
- Нужно: (1) перенос порогов по квантилям: для класса c порог f0 τ_c ↦ доля q_c детекций f0 со score ≥ τ_c на одном
  наборе окон, τ'_c = квантиль 1 − q_c уверенностей final на тех же окнах (либо честная перекалибровка);
  набор окон и правило фиксируются ДО замера; (2) отдельная приёмка bot-10.

## Сборка (только после (1) и (2))
    python make_on.py <zip выкл.> <out.zip> R /workspace/kaggle_out/koz3-dfine-spor-final/dfine_final <thr_final.json>
`thr_final.json` = {"kurgany_povrezhdennye": …, "kurgany_tselye": …, "gorodishcha": …, "arkhitektury": …,
"fortifikatsii": …}. Без него make_on.py с не-f0 весами завершается с ошибкой (проверено 29.10).
Затем check_submission (проба весов на опорном участке 103 тоже должна дать PASS) и сквозной прогон PC.
