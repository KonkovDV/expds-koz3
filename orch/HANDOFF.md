> **06.11.2026 08:00 МСК — LB: YOLO спутника R = 0,38441 (< 0,43274), не берём.** Спутник +14 TP / +324 FP. Лучший по-прежнему v2 + c5-in R (0,43274). Разбор и следующий шаг — `docs/LB_LOG.md`.

> **06.11.2026 02:10 МСК — кандидат v2 + c5-in R + YOLO спутника R готов к LB-пробе.**
> - `v2_c5inR_yoloR.zip`, sha256 `bdfc7f8b631bd0558db03cb7f808d288e38ef8e3a2138913989cb922228c31fe`, 831 557 252 байт.
> - Drive «2» → `00_SUBMIT_v2_c5inR_yoloR_bdfc7f8b`: https://drive.google.com/file/d/17AVG5mzikXYahFAmYq4_RivkjQgY9iqo/view
> - `check_submission`: PASS, FAIL 0. Прогон на 5 участках только со спутником: YOLO заменяет спутник, 121 объект.
> - **Модель:** YOLO26n, обучена на CPU (640 px, 30 эпох, фолды 0–3). Фолд 4 OOF: 0,248 против стека v2 0,146. Порог 0,10 на все классы, форты не выдаются (план §3.10).
> - На LB должен измениться только спутник (база 0,43274, спутник .466/.265). Разбор после пробы — `scripts/lb_diff.py`. Правило: добавленная выдача спутника должна попадать чаще безубыточности, F/(4−F) ≈ 9–10 %.

> **06.11.2026 00:40 МСК — квота GPU Kaggle `dmitrijkv` исчерпана (30 ч/нед).** Ждут сброса: c12-c (f4), ядро ансамбля, обучение YOLO спутника.
> - Автопилоты (`/workspace/autopilot_*.sh`, копии в `orch/autopilots_20261105/`) повторяют отправку каждые 30 мин. Шаги идемпотентны: уже отправленное ядро не дублируется.
> - c12 a/b: Li-only f0–f3 0,657 / 0,491 / 0,406 / 0,247 (c5-in 0,683 / 0,447 / 0,482 / 0,279) — `kaggle/2026-11-05/li_c12_in/PREREG.md`.
> - Контейнер за вечер перезапускался дважды. После перезапуска поднять три автопилота заново.

> **05.11.2026 21:25 МСК — ночные автопилоты (c12, ансамбль, YOLO спутника); оптика по рецепту 3-го места.**
> - **Автопилоты** — копии в `orch/autopilots_20261105/`, рабочие экземпляры и логи в `/workspace/autopilot_*.{sh,log}`:
>   1. `autopilot_c12.sh`: c12-c (f4) после первого COMPLETE из a/b → OOF → гейт `c12-in` с правилом против c5 → `kaggle/2026-11-05/li_c12_in/gate_c12in_ctx.json` → Drive `12_c12in_2026-11-05`.
>   2. `autopilot_ens.sh`: после (1) — ядро `koz3-li-ens-oof` (OOF ансамбля c5 + c12 прод-кодом на T4) → гейт `e512-in` против текущей линии → `kaggle/2026-11-05/li_ens_in/gate_e512in_ctx.json` → Drive `13_ens_2026-11-05`.
>   3. `autopilot_yolo.sh`: датасет `dmitrijkv/koz3-spor-yolo` (910 окон SpOr при 0,5 м, 3928 рамок, 5 фолдов семей) → smoke YOLO26-s → 5 фолдов в рецепте default → 5 фолдов в рецепте GigaFlex (после ансамбля). Выходы — `/workspace/yolo_out/`.
> - **Если контейнер сняли:** перезапустить нужный скрипт с того шага, где оборвался лог (шаги идемпотентны: проверяют статусы ядер).
> - **Оптика:** план `docs/plans/OPTICS_DETECTOR_PLAN_20261105.md`, в том числе §3.9 — OSINT по GigaFlex (`Strekalov/arhipelag`: рецепт ultralytics).
>   Следующий шаг после ядер YOLO: OOF → пороги правилом метрики → прод (вендор ultralytics) → LB-проба K.
> - **Чужие веса:** не подключены. Условие — подтверждение от самой команды (`docs/THIRD_PARTY.md`). Образ `strekalov/rt-core:v2` владелец выгружает на Drive — разобрать только код.

> **05.11.2026 20:40 МСК — LB 0,43274: v2 + c5-in R, новый лучший (+0,0204).** Лидар F1 .510 → .608, остальное как у v2. Разбор — `docs/LB_LOG.md`.
> - **Финал:** эта посылка + вторая лучшая (сейчас v2). Следующий кандидат на второй слот — c12-in R или ансамбль c5 + c12, если пройдут гейт и пробу.
> - **c12:** ядра a/b считаются. Автопилот посчитает гейт и правило c12 − c5. Прод для 12 каналов готов (`3546e54`): подготовка побайтно = вход обучения на 044 и 181.
>   При победе c12: `make_candidate.py --variant R` с весами c12; паритет head-декодирования по образцу `parity_head.py` → `check_submission` → проба.

> **05.11.2026 20:25 МСК — FR1h на LB 0,41112 (< v2): не берём. Следующая проба — v2 + c5-in R. c12 a/b считаются, автопилот гейта запущен.**
> - **FR1h:** изменился только спутник, −3 TP / −17 FP, итог −0,00126. Разбор — `docs/LB_LOG.md`. Финал по-прежнему v2.
>   Без явной загрузки финала платформа возьмёт две лучшие публичные: v2 (0,41238) и FR1h (0,41112).
> - **Проба v2 + c5-in R** (sha256 `98e0ab74…`, ссылка ниже) — следующая. Если она выше v2, финал = она + v2.
> - **c12:** первые a/b упали (Kaggle распаковал zip без `cnx_relief/`), исправлено в `3b79e30`. Smoke OK; a/b (версия 2) идут с 20:06.
>   Автопилот `/workspace/autopilot_c12.sh` (лог `/workspace/autopilot_c12.log`): c (f4) → гейт с правилом c12 − c5 → JSON в ветку → Drive.
>   Если контейнер сняли: шаги — в `kaggle/2026-11-05/li_c12_in/PREREG.md`, «Ход».
> - **Сборщик кандидата:** `make_candidate.py` получил `--base-config` и сам кладёт колёса `inference/cnx/wheels` (в git их нет) со сверкой sha256.

> **05.11.2026 19:55 МСК — гейт c5-in: R PASS; кандидат v2 + c5-in R на Drive, ждёт LB-пробы; c12 загружается на Kaggle.**
> - **Гейт c5-in** — `kaggle/2026-11-05/li_cnx_in/gate_cnxin_ctx.json`:
>   - R (заменить курганы li, li_coarse, li_mosaic): Δ+0,0118, P 0,983, new34 +0,026 — PASS;
>   - K: P 0,897 — FAIL.
>   Это первая линия лидара, которая прошла гейт.
> - **Кандидат** `v2_c5in_R.zip`, собран `make_candidate.py --variant R`:
>   - sha256 `98e0ab74a97fc0b18798b026adc51d96af6891d01ee1edd64432429278eed5ac`, 822 757 161 байт;
>   - `check_submission`: PASS, FAIL 0; smoke — 61 с на 1 LAS;
>   - Drive «2» → `00_SUBMIT_v2_c5in_R_98e0ab74`, md5 OK. Прямая ссылка: https://drive.usercontent.google.com/download?id=1PzJe2Ce42IXCV_My91exfG_oOUQBHaqv&export=download&confirm=t
>
>   LB-пробу отправляет владелец. Должны измениться только P/R лидара (на LB у v2 было .516/.504).
> - **FR1h** (`fe8d60b9…`) на LB ещё не был. Прямая ссылка: https://drive.usercontent.google.com/download?id=1b3lBSIE5TqfjzsIBw3pkwB_9GPhrBnn8&export=download&confirm=t
> - **c12-in.** 12 каналов на 59 LAS готовы; исправления имён и участка 187 описаны в PREREG, раздел «Ход».
>   - Приватный `dmitrijkv/koz3-cnx-li-prep12` готов. Ядра `dmitrijkv/koz3-li-c12-in-a` (f0, f1) и `-b` (f2, f3) отправлены в 19:56.
>   - Ядро c (`c12-in:4`) — вручную, когда освободится слот GPU.
> - **Копия c5-in на Drive:** вывод ядра f4 (`koz3-li-cnx-in-c`) и `gate_cnxin_ctx.json` → `11_c5in_2026-11-05`, md5 OK.

> **05.11.2026 16:25 МСК — вторая сессия: аудит, паритет прода, автопилот гейта c5-in и c12.**
> - **Аудит** — `docs/AUDIT_2026-11-05.md`:
>   - финал v2 сверен (sha256, пересборка); `main` + артефакты v2 даёт побайтно тот же ответ на 5 участках S3;
>   - риски: код EN в финале, папка Drive «2» открыта по ссылке — закрыть.
> - **План к 0,61** — `docs/plans/PLAN_TO_061_20261105.md`.
> - **c5-in.** f4 = `dmitrijkv/koz3-li-cnx-in-c`, отправлено в 15:07. Прод-декодирование «как в ядре» (`[li_cnx] decode = "head"`): паритет PASS на 5 участках.
> - **Автопилот** `/workspace/autopilot.sh`, лог `/workspace/autopilot.log`:
>   - ждёт f4, считает `gate_cnxin_ctx.py 2000` и пушит `kaggle/2026-11-05/li_cnx_in/gate_cnxin_ctx.json` в эту ветку;
>   - затем собирает приватный `dmitrijkv/koz3-cnx-li-prep12` и отправляет ядра c12-a/b.
> - **Если автопилот не дошёл** (контейнер снят): бокс с Drive по `scripts/box_restore.sh`, шапка; затем `gate_cnxin_ctx.py check` и `2000`.
> - **При PASS:** `kaggle/2026-11-05/li_cnx_in/make_candidate.py --variant K|R` → `check_submission` → LB-проба владельцем. Пробная сборка 822 МБ отработала на 181 и 044.
> - **c12-c** (`c12-in:4`) отправить вручную после a или b.
> - **Токены** Kaggle и ключи S3 были в чате: Kaggle-токен перевыпустить.

> **05.11.2026 10:40 МСК — СТОП по просьбе владельца (лимиты).**
> - **Идут на Kaggle:** `dmitrijkv/koz3-li-cnx-in-a` (f0, f1) и `koz3-li-cnx-in-b` (f2, f3), c5-in, запущены в 10:37.
> - **Не отправлено:** `koz3-li-cnx-in-c` (f4). Команда — в `kaggle/2026-11-05/li_cnx_in/PREREG.md`, раздел «Ход».
> - **Дальше:** скачать OOF, `scripts/box_restore.sh`, затем `gate_cnxin_ctx.py check` и `gate_cnxin_ctx.py 2000`.
> - **Токены Kaggle,** вставленные в чат 05.11, стоит перевыпустить: в git их нет.

> **05.11.2026 10:20 МСК — лидар c5-in готов к запуску; второй слот финала — после LB-пробы FR1h.**
> - **Лидар c5-in (ConvNeXt-T, ImageNet).** `kaggle/2026-11-05/li_cnx_in/`: PREREG закоммичен до запуска, ядро собирается без S3, гейт K/R.
>   Запуск: 3 ядра на `dmitrijkv`, около 8 GPU-ч. Нужен токен Kaggle `dmitrijkv` — в облачной сессии 05.11 его не было.
> - **Соседский приор из плана EN снят.** Он уже мерился 29.10: REJECT.
> - **Почему E1_G1 ниже v2.** Разбор в целых числах — `analysis/2026-11-05/lb_diff/README.md`: потеря E1_G1 — находки двух карт MagGr теста, рабочая точка слишком осторожна.
> - **Финал (ТР 8.3.2, 8.3.6).** Засчитывается лучшая из двух попыток на закрытой выборке. Без явной загрузки платформа возьмёт v2 и E1_G1.
>   Предложение: до 14.11 отправить на LB v2+FR1h (`submission_fr1h_on_eb268d6.zip`, sha256 `fe8d60b9…`; на LB не был).
>   Если он ≥ v2, второй слот — FR1h; иначе — E1_G1.
> - **Конкуренты 05.11** — `docs/COMPETITORS_ATOMIC.md`, раздел «Проход 05.11».

> **04.11.2026 21:05 МСК — следующий шаг: лидар как у EN (ImageNet-инициализация ConvNeXt, 9–12 каналов, соседский приор).** Читать `docs/plans/EN_LIDAR_RECIPE_20261104.md` первым. Финал — v2 (`00_SUBMIT_FINAL_v2_EN_167478e3`). Токены Google ещё не отозваны.

> **04.11.2026 20:25 МСК — архив финала готов: v2 (EN) пересобран из `eb268d6`.** `submission_v2_eb268d6_rebuild.zip`, sha256 `167478e3…`, Drive «2» → `KOZ3_backup_2026-11-04/00_SUBMIT_FINAL_v2_EN_167478e3`. Члены совпадают с оригиналом (проверка через FR1h, `docs/REPRO_V2.md`). E1_G1 на LB 0,38988 — запасной чистый вариант. Рост выше 0,41238 — только новыми моделями: лидар (33 % теста) или полнота геофизики.

> **04.11.2026 17:00 МСК — владелец разрешил линию EN (вариант B, с декларацией): `docs/PIXGBM_DECISION.md`.** Сначала LB-проба чистой E1_G1; если она ниже v2 — финал на v2 (EN, 0,41238).

> **04.11.2026 16:30 МСК — v4 на площадке 0,34418 (ниже v2 0,41238 и v3a_min 0,37838). v4 больше не отправлять.** Причины по логам трёх посылок на одном тесте: D-FINE на спутнике (точность .456 → .197) и порог ортофото 0,484 (ортофото 0/0); геофизика без модели пикселей (полнота .656 → .311) — с 30.10. **Следующая посылка — `submission_pixgbm_clean_e1g1_p4_lleaves.zip` (sha256 `ed51cae4…`)**: дерево v2 + чистая геофизика E1_G1, без кода EN, спутник и ортофото как у v2; Drive «2» → `KOZ3_backup_2026-11-04/00_SUBMIT_NEXT_e1g1_ed51cae4`. Проверка сабмита — PASS. Разбор: `docs/LB_DROP_20261104.md`.

> **04.11.2026 14:15 МСК — четырёх аккаунтов Kaggle больше нет: `elenakonkova`, `usertest1234`, `aaaa0000004`, `aaaadg4gg4g4g`.** Профиль 404, API отвечает «users.get denied», как для несуществующего пользователя, а их токены сервер не узнаёт (401). 02.11 все работали. **Подтверждено владельцем 04.11:** при входе Kaggle пишет «Your account has been suspended or banned» — подробности в письме на почту аккаунта. Вероятная причина — правило Kaggle «один аккаунт на человека» (несколько аккаунтов из одной автоматики). Работать только на `dmitrijkv`, новых аккаунтов не создавать. Стек базы гейта перенесён в `dmitrijkv/koz3-gate-box-20261104`, `scripts/box_restore.sh` берёт всё с `dmitrijkv`. Потеряны копия v4 на `aaaa0000004` (v4 есть на Drive и пересобирается из `main`) и OOF Faster R-CNN (линия закрыта).

> **04.11.2026 11:45 МСК — `sat_cl2` FAIL (E1 −0,0055, E2 −0,0054, E3 «только архитектура» +0,0030 при P 0,93).** Сеть городищ/архитектуры SpOr до сдачи не открывать. `--copy-paste` лидара до 08.11 честно не проверить (нужен прод-путь по 59 LAS, 39,8 ГБ, в 5 фолдах). Финал прежний — v4 `f0def982…`; главный рычаг — посылка v4 Еленой. Подробно: `kaggle/2026-11-04/sat_cl2/PREREG.md`, `docs/plans/PROMPT_TO_061_20261103.md` §1–1а.

> **04.11.2026 08:25 UTC — резервная копия доков.** Приватный Kaggle-датасет `dmitrijkv/koz3-handoff-20261104`: 12 файлов (HANDOFF_20261104.md, README, STATUS, этот файл, аудит, разбор балла, задание к 0,61, V4_SUBMIT, конкуренты, LB_LOG, SHA256SUMS). Приватность проверена: аноним и три других аккаунта получают 403. Токен `elenakonkova` с 04.11 отвечает 401 (отозван) — ядра запускать с `dmitrijkv`. Веса и архивы не дублировались: они в копиях 02.11 (строки ниже) и на Drive KOZ3-final.

> **03.11.2026 — ОДИН MAIN.** Все ветки и PR сведены в `main` (`docs/AUDIT_2026-11-03.md`). Новые ветки — от `main`, в `main` — через PR; флаги по умолчанию OFF; `geo_pixgbm.py` в `main` не класть. Задание на гейты — `docs/plans/PROMPT_TO_061_20261103.md`: забрать OOF `dmitrijkv/koz3-w2-dfine2-{a,b}` и `elenakonkova/koz3-w1-satdet-frcnn` (complete 02.11), гейт поверх v4. Финал и порядок посылок — без изменений (`HANDOFF_NOW_20261102.md` §2). На Drive в общей папке две сборки sat2 с одинаковыми именами частей — не склеивать вслепую.

> **02.11.2026 18:40 MSK — DEEP FREEZE. Читать первым: `orch/HANDOFF_NOW_20261102.md`** (v4 primary f0def982…, все add-on рычаги 02.11 FAIL, STANDBY, Hold LB — сабмит только Елена).

# HANDOFF KOZ3 — АКТУАЛЬНО 02.11.2026 15:05 MSK (W4 agent 5)

**v4 СОБРАН:** `koz3_v4c1f_07795d2_51882efd_w6dfd2e07.zip` sha256 `f0def982…7202cef`. Drive [v4c1f_f0def982](https://drive.google.com/drive/folders/1h8AAWDV6xJkgxN7FwZXeLMYaAIEUx5Ia). Kaggle DS `aaaa0000004/koz3-w4-v4c1f`. A уведомил Елену сабмитить; боты LB не пушат. Порядок: v4 → rollback v3_on (1406c01a…) → v2clean (545ef32a…).

**W4 сейчас:** независимый перегейт COMP_GORLOC + opt COMP_FL (честный-10 на v4). Hold LB.

---
# HANDOFF KOZ3 — как продолжить работу только по GitHub (обновлено 02.11.2026 14:40 MSK, coord/бот 1, передача №3)

## АКТУАЛЬНО 02.11.2026 14:40 MSK (передача №3, ФИНАЛЬНАЯ: у coord ~8% квоты; боты свёрнуты, leases сняты)
**Финалы прежние:** «смелый» v3_on **b447649** (sha `1406c01a…ef41d`) основной, «надёжный» v2clean (`545ef32a…`) запасной. Оба Kaggle parity PASS. Сабмитит только Елена. *(УСТАРЕЛО 15:05: v4 собран, см. строку 3 и FINAL_CANDIDATES §0.)*

**C1F (центр-приор sp_dfine) — PASS у W4 после исправления правила порогов:**
- W1 FRZ (bots/w1-satq 5cc5ed7): PASS Δ+0.0159 P1.000. W4 независимый перегейт (bots/w4-v3on PREREG e6b5a31 → 5d0dd4c): ~~FAIL Δ+0.0122~~ — **ОТМЕНЁН** (AMEND1 c60e50f8, неверное правило порогов); итог W4 607edef = PASS (ниже).
- Причина расхождения (W1, bots/w1-satq **3929737**, `docs/W1_FRZ_DISCREPANCY.md`): c1f_prep.sweep у W4 подбирал пороги sp_dfine по обычному F1 (FP=1, порог = сам скор), а прод и метрика — FP=0.5 и порог посередине между скорами (metric._f1/_group_cuts). Без приора правило прода воспроизводит 50/50 базовых порогов, правило W4 меняет все 50. С правилом прода Δ+0.0149/+0.0152, new34 ≥0, kt ≥0.
- W4 уже **подтвердил санити** (bots/w4-v3on **c60e50f8**, `docs/bots/2026-11-02/bot-W4/c1fx_regate/PREREG_AMEND1.md`, ident_check.json: старое правило меняет 50/50, прод-правило 0/50) и закоммитил поправку к PREREG ДО замера: варианты C1 (все классы), GA (gor/arkh), FP1 по отдельности, GAFP снят. Замер выполнен — см. РЕЗУЛЬТАТ ниже. Промежуточные цифры в AMEND1 (GA +0.0134, FP1 +0.0086 при промежуточном правиле) — не результаты.
- **РЕЗУЛЬТАТ W4 (bots/w4-v3on 607edef, рецепт в orch/bot-W4/STATUS_W4.md), правило метрики, санити 0/50:** **C1F-all PASS Δ+0.0162 P1.0 new34 +0.0007 (лучший)**; C1F-gor/arkh PASS Δ+0.0134 new34 0; FP1 PASS Δ+0.0086. Расхождение W1/W4 независимо подтверждено.
- **ПЕРВЫЙ ШАГ:** собрать v4 = v3_on b447649 + фикс 98cdf56 + флаг C1F-all → check_submission → Kaggle parity 4+ сред → sha256 → FINAL_CANDIDATES.md → копии GH/Drive (KOZ3-final) + приватный Kaggle → Елене «v4 готов», сабмитит она.
- **Баг:** в b447649/af6b827 scene_drel считается до признаков D-FINE → флаг C1F в архиве no-op. Фикс **98cdf56** (bots/w4-v3on, под флагом, 474 теста pass); флаги off в v3_on и v4_on.
- **FP1** (frame_prior перепривязан к D-FINE): W1 PASS Δ+0.0086 P0.999 (e574579), **подтверждён W4 607edef** (Δ+0.008581 P0.999).
- **Комбинация C1F+FP1 не аддитивна** (W1 предварительно, PREREG 5691184: kt −0.0035) — в v4 не включать; гейтировать только отдельным PREREG.

**FAIL (после 14:00, флаги off):** Q6 GBM находок (3b7a097; гейт не моделирует прод-постпроцессинг maggr и фильтр finds — при повторе нужен новый PREREG); LI-KURG-TAU с заморозкой (e6089c2, d4c550f; все FAIL); Q1 arkh (P 0.974–0.984 < 0.99), FP2 (new34 −0.0007), Q2 ≥4 Мпикс (new34 −0.0013) — e574579. KDUAL не перегейчивали: перевыбора чужих порогов там не было.

**Урок (обязателен):** при гейте (1) замораживать пороги чужих групп на базе (taus_V3P[fold]); (2) подбирать затронутые пороги РОВНО правилом метрики (FP=0.5, порог посередине между скорами, first argmax, round 6); (3) санити до замера — без приёма воспроизвести все базовые пороги (0 изменений) и базу 0.345439 / 0.333787.

**В работе на Kaggle (без присмотра):**
- W1 SATDET Faster R-CNN: кернел elenakonkova/koz3-w1-satdet-frcnn v2, ETA ~17:25 MSK 02.11 → гейт через tools/satdet/run_det.sh (bots/w1-satdet); STATUS_W1 a0c10733. Детали: `orch/bot-W1/STATUS_W1.md`.
- W2 P-W2-13 второй D-FINE (sp_or), кернелы a/b RUNNING на dmitrijkv (STATUS_W2 5b5e3251) (PREREG 1c0ddc0 + 34f97d3, bots/w2-lidar a614dfa): ручной ранний стоп и гейт `gate_dfine2_ctx.py 2000` — шаги в хвосте `orch/bot-W2/STATUS_W2.md` (w2_notes/dfine2/README.md).
- Статусы ботов на 14:12–14:27: `orch/bot-W1..W4/STATUS_W1..W4.md` (копии с бокса; W4 также в bots/w4-v3on orch/bot-W4/STATUS.md).

**Рутина «KOZ3 ночной контроль» ПРИОСТАНОВЛЕНА 02.11 14:27 MSK.** Боты сворачиваются, leases снимаются.
**pool_kaggle.sh** требует `<username> <kaggle args>`.

## АКТУАЛЬНО 02.11.2026 14:05 MSK (передача №2, история)
**Финал без изменений:** «смелый» v3_on **b447649** (sha `1406c01a…ef41d`), «надёжный» v2clean (`545ef32a…`). Оба Kaggle-parity PASS. Сабмитит только Елена: v3_on → при LB < 0.4124 откат на v2clean. Лучший LB 0.41238.

*(история, УСТАРЕЛО: перегейт W4 выполнен — C1F-all PASS 607edef; v4 = v3_on + 98cdf56 + C1F-all, без FP1)*
- C1F (W1, bots/w1-satq): PREREG 1532475 → результат 5cc5ed7 (`docs/RESULTS_W1_FRZ.md`). Приор «центр сцены» для спутника (sp_dfine) при **замороженных на базе порогах всех не-спутниковых групп**: **Δ+0.0159 P1.000**, new34 0.0000, nolidar +0.0285, все классы ≥0 (kt +0.0003) → PASS. P2F (gor+arkh) Δ+0.0149 PASS (запасной). A1F FAIL (kt −0.0033).
- Финальный конфиг C1F: **af6b827** (`docs/W1_C1F_HANDOFF.md`, `configs/v3_on.toml [post.w1_c1f]/[post.w1_p2f]`, выбор на всех данных d0 0.25, s 0.75; флаги **по умолчанию off**; tests/test_sat_recall.py).
- W1-SATQ (PREREG 0614536 + Amendment 1 d425325, результаты **e574579**, `docs/RESULTS_W1_SATQ.md`): **FP1 PASS** — починка frame_prior (параметры v3_on 0.5/0.5 перепривязаны к sp_dfine, без подбора): **Δ+0.00858 P0.999**, new34 +0.0009, nolidar +0.019, полы ok. Q1a/b/c (arkh-приор/τ) FAIL по P; FP2 nested и Q2 (≥4 Мпикс) FAIL по new34. C1F и FP1 вместе ещё **не гейтились** — нужен совместный гейт.
- W4 перегейт C1F: PREREG **e6b5a31** (bots/w4-v3on, `docs/bots/2026-11-02/bot-W4/c1f_regate/PREREG.md`), слот box/c1f до 19:57. **Найден баг порта af6b827:** в проде `_sp_dfine_phase` не вызывает `_with_scene_drel`, поэтому в архиве C1F-приор был бы no-op (работали бы только пониженные τ). W4 чинит под флагом (off = побайтно как b447649) + unit-тест. При PASS W4 собирает архив v4 и Kaggle parity (как v5 для b447649), sha256 → FINAL_CANDIDATES.md.
- Оговорка: правило заморозки придумано после просмотра new34 (W4 Q0), так что проверка new34 не полностью слепая — записано в PREREG.

**Урок дня (обязателен для всех гейтов):** при гейте приёма для одной группы **замораживать пороги всех остальных групп на базе** (taus_V3P[fold]). Nested-перевыбор чужих порогов (лидарные gor: gor_phys f4, li_coarse f5/f7) давал ложный минус new34 (W4 Q0 7c33027, `docs/bots/2026-11-02/bot-W4/q0_new34/Q0_NEW34.md`). Именно это превратило C1 (FAIL) в C1F (PASS). Старые «почти PASS» линии стоит перегейтить с заморозкой.

**Закрыто 02.11 после утра (FAIL, флаги off):**
- W1 SATREC2 A1–A3/B1–B3 (3dcd07e); W1 SATQ Q1a/b/c, FP2, Q2 (e574579); W1 SATDET трек P1/P2 (PREREG 843d807) — см. RESULTS в bots/w1-satq / bots/w1-satdet.
- W2 P-W2-12 мягкий бленд сидов лидара (PREREG d338318, результат 3dbe4ce): blend3 Δ−0.0035, blend4 Δ−0.0003, целые курганы −0.015.
- W3 KURG-SUBCLS (30c12b3, AUC 0.689), KURG-SUBDROP (3c92fd7); Q3 gor-swap Δ+0.0031 P0.787 new34 −0.0076, Q4 li→sp_or Δ+0.0006, Q5 спутн. хедж Δ−0.0003 kt −0.0063 (bots/w3-kurgan-seeds 1028b70, 8af2969, f7e5545).
- W4 Q6 GBM находок (clean room; PREREG 870433c, A1 a4da8a9, 3b7a097): прод-реплей все FAIL (UNI Δ−0.0075, REP −0.016), хотя new34 у UNI +0.02 — разрыв gate↔прод из-за merge_within_markup/filter_finds_points_regimes. Детали `docs/bots/2026-11-02/bot-W4/q6_finds_gbm/HANDOFF.md`.

**В работе (на 14:05 MSK):**
- Бот 2 (W1): SATDET Faster R-CNN R50-FPN v2 OOF на GPU **elenakonkova** (ETA ~17:25 MSK) → гейт D1/D2 (tools/satq, run_det.sh, bots/w1-satdet 7cbf4e4).
- Бот 3 (W2): P-W2-13 **второй D-FINE (sp_or)** на GPU **dmitrijkv**, 2 кернела (PREREG 1c0ddc0 + amend 34f97d3, bots/w2-lidar a614dfa), аренды acc2/gpu-1 до 17:48, gpu-2 до 18:48; ранний стоп f0+f3, затем гейт поверх v3_on.
- Бот 4 (W3): перегейт KDUAL / LI-KURG-TAU с замороженными чужими порогами (slot box/q2 до 15:54); Q6 передан от W4 (продолжать только при идее, как закрыть разрыв gate↔прод).
- Бот 5 (W4): независимый перегейт C1F (+ фикс scene_drel) → v4 архив + Kaggle parity.

**Баг:** frame_prior в v3_on b447649 фактически выключен (привязан к спутниковым группам, которые D-FINE выкидывает). Сабмиту не вредит; починка = FP1 (PASS выше).

**Разбор конкурентов:** `orch/COMPETITOR_GAPS_2026-11-02.md` (d2fe1de; Drive 11K6QztmfE40YJJHgCtDfF2BxfCb3fMIC) — 10 разрывов vs EN (0.502), только clean-room спецификации.

**Резервные копии №2:** Drive KOZ3-final/backup_2026-11-02b, GitHub bots/backup-2026-11-02b, Kaggle приватный `<acc>/koz3-handoff-20261102b` на 5 аккаунтах — см. `orch/BACKUP_2026-11-02b.md`. Утренний пакет (веса, results) — `orch/BACKUP_2026-11-02.md`, не дублируется.


## 02.11.2026 утро 09:45 (передача №1, история)
**Финал (не меняется):**
- «смелый» = **v3_on b447649** `koz3_v3on_b447649_6d4764f8_w6dfd2e07.zip`, sha256 `1406c01a9aaafc996862a14f12cde14f6bf05b313ad0bbce14428df1d10ef41d`, 722 697 913 B. honest-10 0.333787 (Δ+0.0198 P0.976 vs v2clean). Kaggle parity v5 PASS 4/4 среды (D-FINE грузится, V3_DEGRADED нет), вывод побайтно = старому ebc8e4ba.
- «надёжный» = **v2clean 545ef32a** `koz3_v2clean_70d335a_c028f4eb_wb042594f.zip`, sha256 `545ef32a70b4994f4d694828ae5eb28f27c9161d7f19a96f5a9d3833b87e093f`, 568 488 239 B. Kaggle parity v5 PASS 5/5 сред (PREREG 8844d49, W4 9e711f6), вывод = REF_check4.
- Оба parity-проверены. **Сабмитит только Елена:** сначала v3_on; если LB < 0.4124 → финал v2clean. Старый ebc8e4ba НЕ сабмитить (молча откатывался на v2).
- Где лежат: Drive KOZ3-final (приватная) — папки v3_on_b447649_1406c01a, v2clean_545ef32a, backup_2026-11-02; Kaggle приватный датасет `<acc>/koz3-handoff-20261102` (финальные zip + веса/результаты + orch-доки + SHA256SUMS; на всех 5: dmitrijkv, elenakonkova, usertest1234, aaaa0000004, aaaadg4gg4g4g — приватность проверена 02.11 ~09:55 MSK: чужой аккаунт 403, аноним 403; zip лежат как `.zip.bin`, т.к. Kaggle распаковывает .zip/.tar).

**Гейты, закрытые 02.11 как FAIL (флаги off):**
- W1 SATREC R1/R2/R3/C1/C2 — лучший C1 (приор «центр сцены») Δ+0.0142 P0.9995, режется только new34 −0.0011 (пол ≥0); R1 (единый τ) Δ+0.0100 P0.997, new34 −0.0023. Итог: `docs/RESULTS_W1_SATREC.md` (bots/w1-satrec 06ef4ec). 65% потерь спутника — пороги, не детектор.
- W3 KDUAL (двойная выдача курганов) Δ+0.0031 P0.87 — PREREG 6e1d33d, результат dae47a7, runs/gate_KDUAL.json.
- W3 KDUAL-damaged-only Δ+0.0023 P0.968, new34 −0.0007 — PREREG 5a804b9, b2a5a4d.
- W3 LI-KURG-TAU (4 варианта, Бонферрони P≥0.975): A Δ+0.0044 P0.81; B1 −0.060; B2 −0.055; C +0.0015 — PREREG c96a0d6, результаты 0b35b80 (runs/gate_LIKT.json). Вывод: рост damaged = потеря intact → рычаг в разделении подклассов, не в порогах.

**Сейчас идут (box слоты q1/q2):**
- W1 **SATREC2** (бот 2, ветка bots/w1-satrec2, wt/w1sr): C1+/R1+ с силой приора по классам, расширенная сетка, заранее заданная защита new34; новый PREREG.
- W3 **KURG-SUBCLS** (бот 4, bots/w3-kurgan-seeds): LightGBM-разделение damaged/intact по геоморфометрическим признакам.
- Oracle-gap разбор W3: `runs/gap_breakdown.md` / `.json` (bots/w3-kurgan-seeds b2a5a4d). Потолок 0.690 при базе 0.3220; нет кандидата +0.173, ниже порога +0.173, не тот класс +0.095, геометрия +0.042.
- Конкуренты: `COMPETITORS_2026-11-02.md` (EN 0.502 vs наш 0.412; ⅔ отставания — полнота спутника).

**Безопасность (02.11):**
- 13 публичных датасетов на dmitrijkv **удалены** (xfer s000–s009, gt-train-v2, fr1h-wheels, fr1h-chk-20261101); с другого аккаунта не видны. Все 105 ноутбуков на 5 аккаунтах приватные.
- Drive-папка **KOZ3-pixgbm всё ещё открыта по ссылке** — закрыть может только Елена вручную (коннектор не снимает доступ).
- Ключ S3 организаторов захардкожен в коде 5 **приватных** кернелов (публично не виден; сменить могут только организаторы).
- 31 git-конфиг вне KOZ3 содержит GitHub-токен — ждём решения Елены.
- Kaggle-токены лежат только в `~/.secrets/kpool/<user>` на боксе, **никогда не в git/датасетах**. Вызов: `env -u KAGGLE_USERNAME -u KAGGLE_KEY KAGGLE_API_TOKEN=$(cat ~/.secrets/kpool/<user>) ~/.local/bin/kaggle …` (токены не печатать).
- **ПРАВИЛО: любой Kaggle-датасет создавать только с `"isPrivate": true` в dataset-metadata.json.** Публичный датасет нельзя сделать приватным (API отвечает «успешно», но не меняет) — только удалить. После создания проверять с другого аккаунта (`datasets files` должен падать).

**Резервные копии:** Drive KOZ3-final/backup_2026-11-02 (27 частей, 2.4 ГБ, BACKUP_MANIFEST.md с sha256 и инструкцией восстановления); GitHub bots/backup-2026-11-02 @ 78ffd05 (947 текстовых файлов); этот манифест — `BACKUP_2026-11-02.md`.

## История (до 02.11)
> **ФИНАЛ 01.11 23:50 MSK:** «смелый» = **v3_on b447649** (sha 1406c01a…ef41d, 722 697 913 B, Kaggle parity v5 PASS 4/4 среды, вывод побайтно = ebc8e4ba; Drive KOZ3-final/v3_on_b447649_1406c01a), заменяет ebc8e4ba; «надёжный» = v2clean 545ef32a (Kaggle parity v5 PASS 5/5 сред, PREREG 8844d49, W4 9e711f6). План: Елена сабмитит v3_on b447649 → при LB < 0.4124 откат на v2clean.
> *(прежняя запись)* **ФИНАЛ 01.11 22:35 MSK:** «смелый» v3_on (sha ebc8e4ba…, Δ+0.0198 P0.976, прод Δ+0.0194), «надёжный» v2clean (sha 545ef32a…). Все sha, ссылки на Drive (KOZ3-final), закрытые линии и риск regex — в [`FINAL_CANDIDATES.md`](FINAL_CANDIDATES.md). План: фикс regex (бот 5) → сабмит v3_on Еленой → при LB < 0.4124 откат на v2clean.
> **АКТУАЛЬНО 01.11 18:47:** см. [`HANDOFF_NOW_20261101.md`](HANDOFF_NOW_20261101.md) (живой Kaggle + Drive + next steps).

> Правило: HANDOFF.md обновляется после КАЖДОГО значимого шага (гейт PASS/FAIL, новый прогон, смена железа, решение Елены).

## Цель
- Соревнование KOZ3 (археология: оптика + лидар + геофизика). Цель LB **>0.5**, stretch **0.61**.
- Лучший официальный LB: **0.4124** (29.10, v2 eb268d6). Внутренняя база honest-10 **A=0.314615**.
- Дедлайн сдачи **11.11.2026**. Все новые гейты должны пройти **до 08.11**; 08–11.11 — сборка, check_submission, документы.
- Репо: `KonkovDV/expds-koz3`. Координация: ветка `bots/orch-protocol`, папка `orch/`. **main не пушить никогда.**

## Схема оркестрации (боты)
| Бот | Роль | Ветки |
|---|---|---|
| 1 | coord: план, гейт-решения, очередь, аренды, железо, сводка Елене | bots/orch-protocol |
| 2 | W1: D-FINE / оптика (ae, or, облачные признаки CI2-1) | bots/w1-aeor |
| 3 | W2: лидар (li_*, heatmap lihm, r50c5, CI2-3/CI2-8) | bots/w2-lidar |
| 4 | W3: курганы / городища (сиды, WS-G1) | bots/w3-kurgan-seeds |
| 5 | W4: гейт-перепроверки, геофизика (ORG3), архив, DOC-1 | bots/w4-gate, bots/w4-pix-clean, bots/w4-aidecl |
| helper coord | W5: rerank / пороги | bots/w5-rerank |
Боты пишут статус в `orch/bot-WN/STATUS.md`; coord сводит в STATE.md и COORD_LOG.md. Сообщения бот↔бот асинхронны.

## Железо
- Kaggle (5 аккаунтов): **GPU/TPU только elenakonkova и dmitrijkv**; CPU-only: usertest1234, aaaa0000004, aaaadg4gg4g4g — **у последних нет интернета/S3** (только приватные датасеты Kaggle; никаких ключей в датасетах).
- dmitrijkv GPU: квота до Сб 31.10 03:00. TPU: кернелы часто висят в очереди Kaggle → писать «TPU не занят, простой из-за очереди Kaggle».
- Бокс (компьютер бота): 8 ядер, слоты `slot:box/qN`.

## Где секреты (только имена, значений здесь нет)
- env: `GITHUB_TOKEN`, `KAGGLE_USERNAME`, `KAGGLE_KEY`, `KAGGLE_KEY_1`, `KAGGLE_KEY_2`, `KAGGLE_API_TOKEN`; конфиг `~/.config/kaggle/`.
- Пул Kaggle-токенов: `~/.secrets/kpool/{dmitrijkv,elenakonkova,usertest1234,aaaa0000004,aaaadg4gg4g4g}` (права 600, не в git).
- S3 организаторов: `/workspace/koz3/s3_keys.tsv` (на боксе, не в git). Датасет с cred.json на aaaadg4gg4g4g удалён 30.10 09:40.
- Перед любым push — secret-scan (паттерны KGAT_, ghp_, github_pat_, AWS-ключи). Скрипты кернелов с токенами не пушить, сначала вынести в env.

## Протокол аренд / очереди
- `orch/lease.sh` (take/release/list) → LEASES.json; `orch/lease.sh sync` зеркалит LEASES.json+COORD_LOG.md в bots/orch-protocol. Аренда coord `branch:bots/orch-protocol` до 13.11 — не сокращать.
- `QUEUE.json` + `enqueue.py`: задачи со статусами RUNNING/hold/queued; hold снимает только coord.
- `watcher.py` / `gpu_watcher.py` / `cpu_watcher.sh`: снимают аренды по завершённым кернелам (баг «снимает box-слоты посреди расчёта» исправлен 30.10).
- `COORD_LOG.md`: журнал решений coord (время MSK, что/почему).

## Правила гейта
- honest-10 против A, bootstrap 2000; PASS = **Δ≥+0.003 и P(B>A)≥0.9**, полы по классам не нарушены, new34/nolidar без регрессий.
- PREREG коммитится ДО замера. Модальностный гейт для модальностных сигналов (напр. EtGr только 212–217, остальные байт-в-байт).
- Пост-hoc выбор из k вариантов → **Бонферрони** (P≥1−0.1/k) и независимая перепроверка ботом 5. Пороги после гейта — только nested-fold.

## Жёсткие правила
- Сабмиты на LB делает **только Елена**. Боты не сабмитят.
- Не пушить main. Только рабочие ветки.
- В финале **нет кода, производного от EN** (geo_pixgbm и т.п.). DOC-1 guard: `bots/w4-pix-clean 70d335a`, порт в `bots/w4-gate ab9fd82` — сборка падает при EN-файле. Файлы EN в main/analysis и bots/bot3-v4-patches-nomodels не трогать, в сборку не берутся. Прогоны на EN-коде — только для сравнения.
- Удаления/внешние действия — только с одобрения Елены.

## Рутина «KOZ3 ночной контроль» (каждые 15 мин)
Проверяет кернелы всех 5 аккаунтов и бокс, снимает/выдаёт аренды, дозагружает свободные слоты из QUEUE (не hold), пингует простаивающих ботов, пишет в COORD_LOG, Елене — только при приросте или блокере.

## Экономия токенов
Лимит на исходе (обновится Вс 01.11 19:21). Минимум сообщений, без подтверждений-квитанций; при смене аккаунта новый ассистент читает HANDOFF.md → STATE.md → MASTER_PLAN.md → хвост COORD_LOG.md → QUEUE.json/LEASES.json.

## PROMPT для нового ассистента (скопировать целиком; обновлено 02.11.2026 14:40 MSK)
```
Ты продолжаешь проект KOZ3 «Экспедиция DS. Комплексирование» (соревнование по археологии: оптика/спутник + лидар + геофизика)
для Елены Коньковой. Работай на русском, коротко. Цель LB > 0.5 (лучший наш 0.41238; EN 0.502). Дедлайн 11.11.2026,
все новые гейты — до 08.11; 08–11.11 только сборка, check_submission, документы, декларация ИИ.
Елене пиши ТОЛЬКО при приросте балла (PASS гейта / готовый архив) или блокере. Экономь токены.

0. ПРАВИЛА БЕЗОПАСНОСТИ (нарушать нельзя):
   - Никогда не печатать и не коммитить токены/ключи (Kaggle KGAT, GITHUB_TOKEN, ghp_/github_pat_, S3/AKIA, cred.json).
     Ключи только в ~/.secrets на боксе. Перед каждым push/выгрузкой — secret-scan (KGAT_, ghp_, github_pat_, AKIA, aws_secret, cred.json).
   - Сабмиты на LB делает только Елена. main не пушить, без force-push. Перед push брать lease (orch/lease.sh acquire).
   - Ничего не удалять/не перемещать (ветки, PR, файлы, датасеты, ПК Елены) без её явного «да».
   - Не трогать чужие RUNNING/QUEUED кернелы Kaggle.
   - В финале нет кода, производного от EN (DOC-1 guard: сборка падает на EN-файле). Идеи EN — только clean-room спецификации + гейт.
   - Kaggle: любой новый датасет ТОЛЬКО с "isPrivate": true в dataset-metadata.json (без поля библиотека делает его публичным;
     public→private Kaggle не умеет). После создания проверить с токена ДРУГОГО аккаунта: `datasets files` → 403/404.
     Вызов: env -u KAGGLE_USERNAME -u KAGGLE_KEY KAGGLE_API_TOKEN=$(cat ~/.secrets/kpool/<user>) ~/.local/bin/kaggle ...
     GPU только elenakonkova и dmitrijkv; usertest1234, aaaa0000004, aaaadg4gg4g4g — CPU без интернета/S3.
   - Drive: всё в приватной папке KOZ3-final (1GzW2XGWIMVybFV2pP3WLryeIIhQeN8NB); доступ не расширять, новые файлы не anyone-with-link.

1. Клонируй KonkovDV/expds-koz3 (приватный), ветка bots/orch-protocol. Читай по порядку: orch/HANDOFF.md (раздел
   «АКТУАЛЬНО 02.11 14:40») → orch/STATE.md → orch/FINAL_CANDIDATES.md → orch/COMPETITOR_GAPS_2026-11-02.md
   → хвост orch/COORD_LOG.md → orch/QUEUE.json, orch/LEASES.json → orch/BACKUP_2026-11-02.md и BACKUP_2026-11-02b.md.
   Бокс: /workspace/koz3 (orch/, wt/coord-orch — worktree bots/orch-protocol, w4geo/dist — архивы).

2. ФИНАЛЫ: «смелый» v3_on b447649 (koz3_v3on_b447649_6d4764f8_w6dfd2e07.zip, sha256 1406c01a…ef41d, 722 697 913 B),
   «надёжный» v2clean (koz3_v2clean_70d335a_c028f4eb_wb042594f.zip, sha256 545ef32a…). Оба Kaggle parity PASS.
   Елена: сначала v3_on; LB < 0.4124 → v2clean. ebc8e4ba НЕ сабмитить. Проверь в OFFICIAL_SCORES.md, сабмитила ли она.
   v4 СОБРАН (07795d2, sha f0def982…): v3_on + 98cdf56 + C1F-all; гейт повторён побайтно 02.11 15:08.

3. АРТЕФАКТЫ: Drive KOZ3-final (v3_on_b447649_1406c01a, v2clean_545ef32a, backup_2026-11-02, backup_2026-11-02b, reports);
   GitHub bots/backup-2026-11-02 и bots/backup-2026-11-02b; приватные Kaggle-датасеты <acc>/koz3-handoff-20261102 и
   <acc>/koz3-handoff-20261102b на всех 5 аккаунтах (zip лежат как .zip.bin; сверяй SHA256SUMS).
   Восстановление бокса: см. BACKUP_2026-11-02.md (cat partNN > tar, sha256sum, tar -xf в /workspace/koz3);
   сырые данные организаторов — из S3 (ключи у Елены, в репо их нет).

4. ПРАВИЛА ГЕЙТА: honest-10 поверх v3_on b447649 (прод-база 0.345439, гейт-база 0.333787), bootstrap 2000 (site-family),
   PASS = Δ≥+0.003 и P≥0.9 (k вариантов → Бонферрони P≥1−0.1/k), полы по классам, new34≥0 и nolidar≥0.
   PREREG коммитится ДО замера; выбор параметров только nested; PASS проверяет независимо бот 5 (W4), затем прод-реплей
   и Kaggle parity. НОВОЕ ПРАВИЛО: при гейте приёма для одной группы ЗАМОРАЖИВАТЬ пороги всех остальных групп на базе
   (taus_V3P[fold]) — иначе перевыбор лидарных gor-порогов даёт ложный минус new34 (W4 Q0, bots/w4-v3on 7c33027).
   Затронутые пороги подбирать РОВНО правилом метрики: FP=0.5, порог посередине между скорами (metric._f1/_group_cuts).
   Санити до замера (в PREREG): без приёма 0 изменений базовых порогов и база 0.345439/0.333787.

5. СЛЕДУЮЩИЕ ШАГИ ПО ПРИОРИТЕТУ:
   5.1 ПЕРВЫЙ ШАГ — собрать v4. W4 (bots/w4-v3on 607edef, правило метрики, санити 0/50): C1F-all PASS Δ+0.0162 P1.0
       new34 +0.0007 (лучший); C1F-gor/arkh PASS Δ+0.0134; FP1 PASS Δ+0.0086. v4 = v3_on b447649 + фикс scene_drel 98cdf56
       + флаг C1F-all (параметры на всех данных по рецепту в orch/bot-W4/STATUS_W4.md). check_submission, Kaggle parity в 4+
       средах, sha256 → FINAL_CANDIDATES.md, копии GH + Drive KOZ3-final + приватный Kaggle-датасет. Затем Елене: «v4 готов
       к сабмиту» (сабмитит она; при LB хуже 0.41238 — откат на v3_on/v2clean). C1F+FP1 — только отдельный гейт/PREREG.
   5.2 Дождаться и гейтировать GPU-линии: W1 SATDET elenakonkova/koz3-w1-satdet-frcnn v2 (ETA ~17:25 02.11) → tools/satdet/run_det.sh;
       W2 P-W2-13 второй D-FINE (dmitrijkv, a/b RUNNING) → ранний стоп по orch/bot-W2/STATUS_W2.md → гейт поверх v4/v3_on.
   5.3 Перегейт «почти PASS» с заморозкой УЖЕ сделан: LI-KURG-TAU FAIL (e6089c2, d4c550f), KDUAL не нужен (перевыбора не было).
       Остаётся по желанию: Q3 gor-swap (+0.0031) с правилом метрики.
   5.4 Остальное из COMPETITOR_GAPS (только clean-room): fold-ансамбль D-FINE на инференсе (+0.003…0.010, гейтом не меряется —
       только с согласия Елены), framing gate на больших сценах в форме без потерь new34, Q6 GBM находок — лишь если
       есть способ закрыть разрыв gate↔прод (merge_within_markup / filter_finds_points_regimes).
   Не повторять закрытое: SATREC/SATREC2, KURG-SUBCLS/SUBDROP, бленд сидов лидара, lidar-aug, r50c5, J2/CTXc, Q1 arkh, FP2, Q2,
   Q4, Q5, Q6 (без нового PREREG с maggr/фильтром finds), LI-KURG-TAU, C1F+FP1.

6. ЧТО У КАКОГО БОТА (на 02.11 14:05): бот 2 = W1 (спутник/оптика, bots/w1-satq, bots/w1-satdet), бот 3 = W2 (лидар +
   второй D-FINE, bots/w2-lidar), бот 4 = W3 (курганы/городища, bots/w3-kurgan-seeds), бот 5 = W4 (гейт, архив,
   bots/w4-v3on). Все боты СВЁРНУТЫ 02.11 ~14:30, leases сняты. Статусы: orch/bot-WN/STATUS_WN.md. Рутина «KOZ3 ночной
   контроль» ПРИОСТАНОВЛЕНА 02.11 14:27 MSK (включать заново только по решению Елены). pool_kaggle.sh: `<username> <kaggle args>`.

7. ОТКРЫТЫЕ ВОПРОСЫ К ЕЛЕНЕ: (a) закрыть доступ по ссылке к Drive-папке KOZ3-pixgbm (коннектор не умеет);
   (b) GitHub-токен в 31 git-конфиге вне KOZ3 — чистить ли; (c) ключ S3 организаторов захардкожен в 5 приватных кернелах;
   (d) ФИО/роли/ИИ-инструменты для декларации ИИ до 11.11; (e) чистить ли общий клон /workspace/koz3/repo
   (81 изменённый отслеживаемый файл, check_conflicts ERROR); (f) рутину с 15 мин на раз в час для экономии квоты.

8. После каждого значимого шага обновляй orch/HANDOFF.md и STATE.md в /workspace/koz3/wt/coord-orch
   (git -c user.name=coord-lease -c user.email=coord@koz3.local commit, lease → push в bots/orch-protocol),
   затем cd /workspace/koz3/orch && ./lease.sh sync.
```
