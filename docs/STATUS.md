# STATUS (KOZ-3) — формат: время MSK | бот | что | число | ссылка

29.10 08:47 MSK | W4 | лучший официальный результат: v2 eb268d6 | 0.41238 | docs/LB_LOG.md
29.10 22:15 MSK | W4 | ветка bots/w4-gate от bots/bot7-candidate-v3fix-nomodels 0037c0e (v3 + CRS-фикс 008_Герите) | — | bots/w4-gate
29.10 22:15 MSK | W4 | v3 за флагами: в configs/default.toml sp_dfine=false и [post.frame_prior] закомментирован (=выкл); включённая сборка — configs/v3_on.toml (= bot-7 work8/default_v3on.toml) через build_submission_git.py --config | — | configs/v3_on.toml
29.10 22:15 MSK | W4 | гейт D-FINE R (SpOr, f0) против A, honest-10 fam10 (bot-9 paired_5f, проверено bot-10) | +0.019802, P 0.976, LOFO +0.023935, 5/5, new34 +0.008397 | docs/bots/2026-10-29/bot-10/REPORT10_accept.md §1
29.10 22:15 MSK | W4 | гейт FR1h, TOFR−TO (БЕЗ D-FINE R) | +0.004042, P 0.999, LOFO +0.004228, 5/5, new34 +0.00044 | docs/bots/2026-10-29/bot-10/REPORT11_audit.md C3h
29.10 22:15 MSK | W4 | ВНИМАНИЕ: при R строки sp_or/sp_focal/sp_focal2 вне 132 SpOr-участков = 0 ⇒ на трейне FR1h поверх R даёт 0; ожидание v3 ≈ +0.0198 к A, не +0.0239 | 0 | docs/bots/2026-10-29/bot-10/PREREG_aeor.md §1
29.10 22:15 MSK | W4 | перезамер v3 одним пакетом на коробке невозможен: нет table.pkl / OOF-стека A (F:/expds3/work на ПК, ПК офлайн) | — | orch/bot-W4/STATUS.md
29.10 22:15 MSK | W4 | v3 e2e на ПК (pc_e2e_v3.ps1) и замер времени 4090 — не сделаны | — | docs/STATUS_2026-10-29.md

## 29.10 22:30 MSK — W4: PIXGBM и патчи bot-3

* PREREG: `docs/PREREG_w4_pixgbm.md` (до замеров). `koz3-w4-pixgbm-s3` = P-retrain (EN-линия), не clean room.
* Патч de1edc9 (ground_metres), флаг выкл: read_band побайтно совпадает с v3 (`bots/w4-gate` 8a11895) на 12 синтетических
  растрах × 5 режимов (EPSG:3857 / 32636 / 32637 / без CRS; 1/3/4 канала): sha256 выхода 9fcb0f24… у обоих; с EXPDS_GROUND_M=1 —
  иное (e9e00bdc…), как задумано. Тесты: base 15F/386P, patched 16F/388P; лишний F — test_load_map_3857_ground_gsd
  (нужна часть патча в geo_pixgbm.py, не применялась). Остальные падения одинаковы (нет моделей/модулей на коробке).
* Части de1edc9 и 1327e88 в `expds/geo_pixgbm.py` W4 НЕ открывал (гигиена clean room): побайтно не проверены.
  1327e88 по коммиту активен только при replace=true и непустом union_name_re; в v3 `union_name_re = ""` (default.toml:291).
* Сквозной побайтный прогон на реальных участках невозможен: на коробке нет растров участков.

### Журнал копий данных (для DATA_PROTOCOL)
| Дата | Где | Что | Объём | Доступ | Удаление |
|---|---|---|---|---|---|
| 29.10 | Kaggle dmitrijkv, ядро koz3-w4-pixgbm-s3 (приватное, /kaggle/working и /tmp) | геофизика 200–217 из S3 | ~1,4 ГиБ (+разметка) | команда | по окончании сессии ядра; вывод — без сырья проверить |

30.10 00:45 MSK | W4 | GEO-ORG1 (pix_org, org_maggr=false) 31a66bc: C-et+ORG1 против C-et, s0, 17 уч. без 200: both Δ+0.0017 (finds +0.042, poly −0.025), pts Δ+0.0042 P_fam≥0.95 2/5, poly Δ−0.0025 → ORG1 сам по себе НЕ принят (порог +0.009). Пакет C-et+ORG1both против A: Δ+0.0204, P_fam≥0.95, 4/5, срезы OK → прокси-гейт ДА (1 сид, без 200) | FAIL (ORG1) / прокси OK (пакет) | /workspace/koz3/w4geo/eval/R17org_vsCet_s0.json, R17org_vsA_s0.json
30.10 00:45 MSK | W4 | GEO-ORG2 (org2_band_mode="jet", org2_mask для EtGr) за флагами, тесты 6/6; замер требует переобучения pix EtGr (новый кэш --band-mode jet --org-mask) — не начат: коробка 16 ГБ занята fit_all s1/s2 | не замерено | 31a66bc
30.10 00:45 MSK | W4 | docs/ai_declaration.md черновик для Елены (п. 8.5.3/13.6 ТР; шаблона организаторов на коробке нет) | черновик | bots/w4-gate 8f527bb
30.10 00:50 MSK | W4 | пакет C-et+ORG1pts против A: Δ+0.0229 (finds +0.042, poly +0.187), P_fam≥0.95, 4/5, срезы OK, F1 не ниже A → прокси-гейт ДА (s0, 17 уч. без 200); pix_org.load без массива расстояний EDT (память) | прокси OK | eval/R17orgpts_vsA_s0.json
30.10 01:05 MSK | W4 | GEO-ORG3 итог s0 vs A17: A+ORG3 +0.0206 (P_fam 0.70 структурно: EtGr только 212–217, 3/5); A+ORG1poly+ORG3 +0.0194 P_fam 0.97 4/5 ГЕЙТ ДА (аналог орг. C_mgpoly_et +0.0206, 5/5); A+ORG1both+ORG3 +0.0236 ДА; C-et+ORG1pts+ORG3 +0.0267 ДА (лучший). Инференс 216: 6.6 с, 153 МБ. Модели full: /workspace/koz3/w4geo/models_orget/org (пороги 0.1/0.1), не в репо | eval/R17orget*_vsA_s0.json

## 30.10 01:35 MSK — W4 v2-clean (bot-5)
- Модальностный гейт C-et+ORG3 (PREREG 5702c78, записан до финала; после прежних замеров — оговорено): 18 уч. vs A18 —
  (a) 12/12 участков без EtGr побайтно = A; (b) Δ +0.022499, p_region(EtGr) 1.0; (c) F1 poly +0.225, finds 0 → **PASS**.
  Совпадает побайтно с ре-гейтом координатора. ORG1-точки исключены (ре-гейт: Судьбищи −0.014).
- Кэш классики 200: `accel.box_mean` поблочно по строкам (побайтно тот же результат), пик 4.7 ГБ вместо OOM.
- Архив v2-clean (НЕ отправлен): koz3_v2clean_8bb5803_76de89fc_wb042594f.zip, sha256 81b48731d53994ebb3211ed68a96eca3d7e215215daea27931c70bcc1767f968,
  58 файлов, configs/v2_clean.toml как default.toml; DOC-1 OK; список файлов — orch/bot-W4/runs/v2clean_files.txt.
- Оговорка о порогах v2-clean: 2 новых порога — `etgr_org` object_poly 0.21579 (ORG3 EtGr) и `terfl_pix` object_poly 0.58017 (TerFl) —
  подобраны ПОСЛЕ гейта, in-sample по OOF C-et+ORG3 на 18 участках (sweep_groups, все участки); в гейте их не было
  (там честные пороги по фолдам), модели в архиве — full, поэтому пороги могут быть слегка занижены.

## 30.10 01:50 MSK — пороги после замечания бота 1 (PREREG 5ea8aea)
- TerFl clean pix выключен (`maps=["etgr"]`), terfl_pix = 0.746421 (default, не используется).
- `etgr_org.object_poly = 0.0` (N0: только внутренний LOSO-порог ORG3). Гейт C'=pix(etgr)+ORG3 vs A18: N0 Δ +0.023191 PASS;
  с честным sweep группы Δ +0.020604 PASS. Пороги гейта по фолдам (sweep): F3 0.09712, F4 0.21579, F5 0.05802.
- 0.21579 / 0.58017 из 99002ab сняты.
- Архив v2-clean 26ad1ba: koz3_v2clean_26ad1ba_0aa67b2b_wb042594f.zip, sha256 15d59e9f7c120b56079f4cc7bc8cb83c14b698db9a57228e75e286dddd9842c9,
  58 файлов, DOC-1 OK. Старый 81b48731… (8bb5803) заменён. (Время PREREG 5ea8aea — по git, «~01:55» в тексте — оценка.)
- check_submission (полный, со смоуком) — Kaggle CPU usertest1234/koz3-w4-v2clean-check, в очереди watcher (w4-v2clean-check).
