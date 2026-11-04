# PREREG C1 — G1 (sat2 только kurgany_povrezhdennye) и G2 (li self_nms@8 только не-курганы) поверх v3_on, прод-контекст

Записано 01.11.2026 (MSK) ДО любого замера G1/G2. Один вариант на гейт, без перенастроек после замера.

## Движок (тот же, что reports/h2h_20261101 на bots/handoff-2026-11-01)
- W4 honest-10: `orch/bot-W4/work/v3_gate_lib.py` → `orch/bot-W1/work/d3_gate.py` (без изменений): TableEngine по `w5data/work/table.pkl`,
  stack_arch + gt_train_v2, family folds (10, site_family), пороги групп пересвипываются на 9 train-фолдах и применяются к hold-фолду
  (пороги/α — только nested внутри train-фолдов). Бутстрэп парный по site_family, 2000, seed 0.
- Санити: V3P на полном пуле = 0.333787 (как в h2h) — печатается, но НЕ является референсом гейта.

## Прод-контекст (steering W1, PREREG_W1_J2NC_CTX на bots/w1-cfdeploy)
В проде постобработка видит только объекты, пережившие ранние полы `output_floors`; гейты на полном пуле давали ложный прирост.
Поэтому оба гейта считаются строго в прод-контексте архива v3_on:
- cfg = `configs/default.toml` из архива `koz3_v3on_b505656_6d4764f8_w6dfd2e07.zip` (== `configs/v3_on.toml@b505656`), полы =
  `expds.pipeline.output_floors(cfg, threshold_group)[class]` из кода ТОГО ЖЕ архива. `w5_cloudfeat` в v3_on нет ⇒ li_floor не
  применяется; `post.kurgan_neighbour_prior` / `post.site_calibration` выключены.
- Референс R = V3P-PC: строки стека без drop3 V3P (etgr; sp_or/sp_focal/sp_arch/sp_focal2 в регионах D-FINE) с conf ≥ пола своей
  группы/класса; добавки C19-etgr и sp_dfine OOF — только с conf ≥ своих полов. Строки ниже пола выкинуты (не участвуют ни в свипе,
  ни в матчинге). V3P-PC печатается рядом с 0.333787.

## G1 — sat2 только для kurgany_povrezhdennye (ДОБАВКА, не REPLACE)
- X = R + OOF sat2 `w1/sat2/pkg/data/sp_det2_oof_notta.geojson` (5 фолдов, без TTA), только class_name = kurgany_povrezhdennye,
  группа sp_det2. Пол sp_det2: kurgany_povrezhdennye = 0.3645 (= пол v3_on sp_or для этого класса; конвенция деплоя combo-sat2:
  `[thresholds_by_markup.sp_det2]` = архивный порог sp_or по курганам). Строки ниже пола выкинуты. kurgany_tselye из sat2 не берём.

## G2 — li-post self_nms@8 только для не-курганных классов
- X = R, в котором к li-семейству (threshold_group или markup начинается с "li") классов gorodishcha, arkhitektury, object_poly,
  fortifikatsii, finds_points применён `li_post.self_nms` (жадный same-class NMS по conf, радиус 8 м, центроиды в UTM-метрах таблицы),
  по региону, ПОСЛЕ прод-полов (в проде li_post стоит после apply_thresholds; полы v3_on = пороги архива). Курганы не трогаются;
  class_tau не применяется. Оговорка: NMS до nested-пересвипа (приближение: пересвип только поднимает порог; межгрупповые li-пары
  при разных порогах — допускаемая неточность).

## Критерий PASS (каждый гейт против R = V3P-PC)
- Δall ≥ +0.003 и P(Δ>0) ≥ 0.975 (бутстрэп 2000 по site_family, seed 0);
- группы: Δ new34 ≥ −0.005, Δ F1 kurgany_povrezhdennye (damaged) ≥ −0.005, Δ F1 fortifikatsii ≥ −0.005;
- дополнительно (только отчёт): полы движка по классам (FLOOR_TIGHT −0.002 / прочие −0.005), nolidar, Δ на полном пуле.
- Один прогон каждого гейта. При PASS — проверка исполнимости в инференсе архива (веса/код в архиве, без сети, без кода EhimenNathan);
  без неё в финал не идёт. Архив v3_on не трогаем.

## Результат 01.11.2026 22:11 MSK (один прогон, tools/c1_gates/c1_gates.py → results/c1_gates/c1_gates.json, 37 с)
- Санити V3P полный пул 0.333787 (совпало). Референс V3P-PC = 0.345852 (прод-контекст; vs полный пул +0.0121 — только диагностика).
  Строк стека в PC: kp 2629, kt 1746, gor 68, fort 38742; добавки 3563 из 86647; sat2 kp ≥ 0.3645: 429.
- G1 (sat2 только kp): Δ +0.001881, P 0.9425; new34 −0.0009, nolidar +0.0052; kp F1 +0.0063, fort 0 ⇒ **FAIL** (Δ < 0.003, P < 0.975).
- G2 (self_nms@8 не-курганы): подавлено 3 строки (gor 2, fort 1); Δ −0.002846, P 0.0045; new34 −0.0060 (FAIL группы), gor F1 −0.019,
  fort +0.0001 ⇒ **FAIL**.
- Ни один гейт не прошёл ⇒ проверка исполнимости в инференсе не требуется; в финал ничего не идёт, v3_on без изменений.
