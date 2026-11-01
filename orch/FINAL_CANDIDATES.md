# FINAL CANDIDATES KOZ3 — 01.11.2026 22:35 MSK (coord)

> Архивы в git НЕ лежат. Копии: Google Drive, приватная папка **KOZ3-final** (не расшарена):
> https://drive.google.com/drive/folders/1GzW2XGWIMVybFV2pP3WLryeIIhQeN8NB
> Склейка: `cat <name>.zip.part0* > <name>.zip && sha256sum -c SHA256SUMS` (см. README.md в подпапке).
> На LB сабмитит только Елена.

## 1. Кандидаты

| Роль | Архив | SHA256 (целый zip) | Размер | Drive | Оценки | Отчёты |
|---|---|---|---|---|---|---|
| **«смелый» (основной)** v3_on | `koz3_v3on_b505656_6d4764f8_w6dfd2e07.zip` | `ebc8e4ba4c583403fd87485e6c394fc98eb1aee24fbb09d4a7d25cbe3e2d9401` | 713 557 985 B (680.5 MiB), 8 частей × 95 MiB | [v3_on_b505656](https://drive.google.com/drive/folders/1r_XrsUo-7gy-uu80T_Dvb5p9oJ5lrExL) | honest-10 **0.333787** vs v2clean 0.313967: **Δ+0.0198, P0.976**; прод-путь (реплей кода архива на OOF): **Δ+0.0194, P0.964** (v2c 0.326009 / v3 0.345439); check_submission **PASS** (FAIL 0, WARN 4) | гейт: [w4-v3on 31630b8 · v3_gate.json](https://github.com/KonkovDV/expds-koz3/blob/31630b8/docs/bots/2026-10-30/bot-W4/v3_gate.json); прод: [w1-cfdeploy 9ea621d · results/v3prod](https://github.com/KonkovDV/expds-koz3/tree/9ea621d/results/v3prod); Kaggle-датасет `aaaa0000004/koz3-w4-dfine-ref103` |
| **«надёжный» (запасной)** v2clean | `koz3_v2clean_70d335a_c028f4eb_wb042594f.zip` | `545ef32a70b4994f4d694828ae5eb28f27c9161d7f19a96f5a9d3833b87e093f` | 568 488 239 B (542.2 MiB), 6 частей × 95 MiB | [v2clean_545ef32a](https://drive.google.com/drive/folders/1omTT3QFxb3zgFAFQjj862Q6LkQ92uBhQ) | honest-10 **0.313967**; DOC-1 guard OK; check_submission PASS (FAIL 0, WARN 2, aaaa0000004/koz3-w4-v2cleanchk2 v4); без D-FINE (риск regex не касается) | — |

- Лучший официальный LB пока **0.4124** (v2 eb268d6, 29.10). **v3_on на LB ещё не отправлялся.**
- Порядок: сабмит v3_on **после** фикса regex (см. §3) → если LB хуже 0.4124, финал = v2clean.
- Отчёты (json/md, маленькие) также в Drive: [KOZ3-final/reports](https://drive.google.com/drive/folders/1-cMktRigtZuZRN7peDVqzJe3YNxZFtnP).
- Не финал: v4_on `koz3_v4on_da537d3…` sha `34819adb…` (J2 FAIL, не включать w5_cloudfeat). `koz3_v3on_10ee67b…` sha `e40b2e8b…` — сборка бота 5 (фикс regex?) в работе, до проверки не использовать.

## 2. Закрытые сегодня (01.11) линии

| Линия | Результат | Где |
|---|---|---|
| J2 / RR1s-cf (cloudfeat реранкер) | nested Δ+0.0117 → архив Δ+0.0052 P0.921 (fort −0.0084); прод-контекст **Δ+0.0032 P0.793**, курганы целые −0.0037 → **FAIL**. Причина: ранжировщик в проде видит только объекты выше output_floors (7738 из ~186k) | [w4-v3on 31630b8 · J2_nocnx](https://github.com/KonkovDV/expds-koz3/blob/31630b8/orch/bot-W4/runs/independent_regate_W1J2_nocnx.json), [v4on_archive_gate](https://github.com/KonkovDV/expds-koz3/blob/31630b8/orch/bot-W4/runs/v4on_archive_gate.json), [w1-cfdeploy 30c6ae1/4c720ef · results/j2nc_ctx](https://github.com/KonkovDV/expds-koz3/tree/9ea621d/results/j2nc_ctx) |
| CTXc (реранкер на прод-контексте, fort выкл) | **Δ−0.0046 P0.124**, new34 −0.024, городища −0.053 → FAIL | w1-cfdeploy 30c6ae1, results/j2nc_ctx/ctxc_variant.json |
| lidar-aug (c7-sc aug) | Li-only F1 mean 0.321 vs 0.360 (**Δ−0.039**; f2 0.343/0.401, f3 0.313/0.361, f4 0.308/0.319) → CLOSED | [w2-lidar 5912dc2 · w2_notes/lidar_aug](https://github.com/KonkovDV/expds-koz3/tree/bdd7e81/w2_notes/lidar_aug) |
| r50c5 (лидар) на V3P, прод-контекст | ранний стоп PASS (0.4395), но гейт **Δ+0.0024 P0.66** (целые −0.0077, гор −0.0151, new34 −0.0132) → FAIL | [w2-lidar bdd7e81 · w2_gate/r50c5_v3p](https://github.com/KonkovDV/expds-koz3/tree/bdd7e81/w2_gate/r50c5_v3p) |
| G1 sat2-kp (sat2 только для повреждённых курганов поверх v3_on) | **Δ+0.0019 P0.94** → FAIL | [c1-gates-1101 d2ce07a · results/c1_gates](https://github.com/KonkovDV/expds-koz3/tree/d2ce07a/results/c1_gates) (PREREG 1eb216d) |
| G2 li nms8 (не-курганы) | **Δ−0.003 P0.0045**, new34 −0.006 → FAIL | там же |
| large-gate (D-FINE ae крупные объекты) | Δ−0.00012 vs A → FAIL | [handoff-2026-11-01 3b2ad52 · reports/h2h_20261101/kernels](https://github.com/KonkovDV/expds-koz3/tree/3b2ad52/reports/h2h_20261101/kernels) |
| ae-dfine-gate | flag_dead (Δ 0.0, ложные детекции) → FAIL | там же |
| v2+sat2(+FR1h) vs v3_on | v3_on 0.3338 vs v2+sat2 0.3236 (**−0.010**); v3_on+sat2 +0.0019 P0.941 → FAIL | [handoff-2026-11-01 3b2ad52 · reports/h2h_20261101](https://github.com/KonkovDV/expds-koz3/tree/3b2ad52/reports/h2h_20261101) (PREREG 879453b) |
| M_rp поверх V3P (compgate) | +0.00276 → FAIL (перекрывается с v3_on) | w4-v3on: docs/bots/2026-10-30/bot-W4/compgate.json |

## 3. Риск regex (D-FINE)

- Если в окружении проверки стоит `regex==2025.11.3`, импорт D-FINE (transformers/tokenizers из vendor) падает, и v3_on **молча** выдаёт результат v2 (теряется весь +0.02). Замечено в одной из Kaggle-проверок.
- Фикс (бот 5): загрузка, устойчивая к версии regex, + громкое предупреждение в лог, если модуль не поднялся. Исправленный архив будет новым sha; этот архив (ebc8e4ba…) **не меняется**.
- До фикса: перед сабмитом убедиться, что в логе инференса D-FINE загружен (нет fallback на v2).
