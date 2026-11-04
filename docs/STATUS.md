# STATUS (KOZ-3) — формат: время MSK | бот | что | число | ссылка

04.11 05:20 MSK | GPU | koz3-sat-cl v2 COMPLETE (55 мин, 5/5 фолдов, 132 участка): OOF 13154 — городища 13140, архитектура 14 (голова архитектуры схлопнулась; f2 учился на 70 позитивных тайлах); гейт D1/D2 запущен | — | kaggle/2026-11-04/sat_cl/PREREG.md

04.11 02:00 MSK | GPU | запущено ядро dmitrijkv/koz3-sat-cl v1: U-Net ConvNeXt-T, маски городищ и архитектуры SpOr, 5 фолдов OOF по полным растрам; PREREG гейта D1 (добавить) / D2 (заменить), P≥0.975 | — | kaggle/2026-11-04/sat_cl/PREREG.md

03.11 23:50 MSK | конкуренты | yakirill99 @e3a803b, лосс: Dice+focal маски, QFL центров, маска валидности класса; проверка по бакету (2132 ключа): пустых файлов разметки нет, расхождения слоёв редки и объяснимы видимостью → маска валидности у нас не применима | — | docs/COMPETITORS_ATOMIC.md

03.11 21:36 MSK | гейт | D-FINE f0 + D-FINE-2 (P-W2-13) поверх V3P_ctx: B Δ+0.0072 P0.916; C Δ+0.0087 P0.956; C2 Δ+0.0123 P0.985; C3 Δ+0.0080 P0.922 — все FAIL по new34 (−0.004…−0.0055); гор/арх растут на старых участках и падают на новых; D-FINE-3 не запускать | Δ≤+0.0123 | w2_gate/dfine2_v3p/PREREG_cls_20261103.md

03.11 23:00 MSK | аудит | один main: 40 веток и PR #16–#30 сведены; исправлены SIGTERM без файла (A1/A2) и тест, писавший код EN в дерево (A3); выход по умолчанию побайтно прежний; на Kaggle не собраны OOF dfine2 (5 фолдов) и frcnn | 0.41238 (LB), v4 0.361641 (honest-10) | docs/AUDIT_2026-11-03.md

31.10 23:55 MSK | ночь | один GPU: доучить spor_focal_r18 с --init-weights и --within-site-rank 1, 4 эпохи, lr 1e-4; без флага старт с ImageNet запрещён | — | docs/plans/PROMPT_WITHIN_SITE_RANK_20261031.md
31.10 23:50 MSK | код | шарнир порядка пиков на кадре, флаг --within-site-rank по умолчанию 0; их скачок до 0.49 — спутниковые головы, не восемь ракурсов и не соседский приор (+0.0003) | — | expds/centers.py within_site_rank_hinge
31.10 23:25 MSK | промт | одна GPU: доучить spor_focal_r18 потерей «на участке верный пик выше громких ложных»; стоп, если медиана ложных сверху не упала с ~18 до ≤2 | — | docs/plans/PROMPT_WITHIN_SITE_RANK_20261031.md
31.10 23:00 MSK | обход | код и журналы: нового ядра нет; 0.41238 уже содержит архивную геофизику и rerank; вне посылки только FR1h и диски 5 см, оба около +0.004 | — | docs/plans/PROMPT_DINOV3_HILLFORT_KAGGLE_20261031.md §12
31.10 22:52 MSK | пилот | peak-rerank STOP все 4 гейта: 25 поднятых, P 0.20, медиана промаха 851 м, train AUC 0.973; GPU не занимать | — | docs/plans/PROMPT_DINOV3_HILLFORT_KAGGLE_20261031.md §11
31.10 22:20 MSK | honest-10 | near_kept FAIL: Δ−0.00188 P0.245 damaged−0.0033 new34−0.00107; 678 кругов ≈42 TP / 636 FP; секцию не включать | 0.3127 | docs/plans/PROMPT_DINOV3_HILLFORT_KAGGLE_20261031.md §10
31.10 22:15 MSK | срез | кольцо 12–80 м вокруг принятого спутникового кургана, пол 0.20: на срезе F1 0.533→0.582 / 0.546→0.609; в default.toml секции нет, честный замер всего стека не снят | — | expds/post.py rescue_near_kept
31.10 22:10 MSK | пилот | sp_dino_gor fold0 числовой PASS впритык (3/15, P 0.33), фолды 1–4 СТОП: сцена заучена, пик мимо полигона | F1 0.182 | docs/plans/PROMPT_DINOV3_HILLFORT_KAGGLE_20261031.md §9
31.10 21:40 MSK | аудит | свой локализатор городищ выключен: секции в default.toml нет, круг в максимуме карты, один сбойный снимок не гасит фазу | — | expds/hillfort_loc.py
31.10 21:16 MSK | владелец | лицензии моделей сняты полностью; рабочий промт локализатора городищ DINOv3 на Kaggle | — | docs/plans/PROMPT_DINOV3_HILLFORT_KAGGLE_20261031.md

31.10 15:24 MSK | A | FR1h [post.frame_prior] honest-10 PASS vs base 0.31462 — не включать в default без «да»; перед zip: thr re-sweep + geo vs 0.769/0.656 | Δ+0.00409 P0.999 damaged+0.0107 | out/fr1h-gate/
31.10 15:08 MSK | W2/A | r50c5 lidar cut honest-10 REJECT — не кандидат, на LB не класть | varb Δ−0.00513 P0.33; varb_kur Δ+0.00088 P0.56; base 0.31462; early-stop f01 PASS mean Li 0.4395 ≠ submit | out/r50c5-gate
31.10 09:10 MSK | аудит | действующая база площадки, не перезаписывать без замера и «да» | 0.41238 | docs/LB_LOG.md
31.10 09:10 MSK | аудит | посылка v3a_min базой не стала | 0.37838 | не загружать снова
31.10 09:10 MSK | аудит | журнал отказов не стирать: это список закрытых шагов. Следующий замер — лидар r50c5; стоп, если повреждённые курганы не растут. Спутник (полнота 0.255) этими ядрами не закрыт | — | bots/orch-protocol e8aa6c8
31.10 09:10 MSK | аудит | dome и slope_gate | false | configs/default.toml
31.10 09:22 MSK | вектор до 11.11 | держать 0.41238. Живой замер только r50c5: стоп, если повреждённые курганы не растут. Спутник этими ядрами не закрыт. Закрытые отчёты — ветка archive/worked-20261031, не кончик main | — | archive/worked-20261031
29.10 08:47 MSK | W4 | лучший официальный результат: v2 eb268d6 | 0.41238 | docs/LB_LOG.md
29.10 22:15 MSK | W4 | ветка bots/w4-gate от bots/bot7-candidate-v3fix-nomodels 0037c0e (v3 + CRS-фикс 008_Герите) | — | bots/w4-gate
29.10 22:15 MSK | W4 | v3 за флагами: в configs/default.toml sp_dfine=false и [post.frame_prior] закомментирован (=выкл); включённая сборка — configs/v3_on.toml (= bot-7 work8/default_v3on.toml) через build_submission_git.py --config | — | configs/v3_on.toml
29.10 22:15 MSK | W4 | гейт D-FINE R (SpOr, f0) против A, honest-10 fam10 (bot-9 paired_5f, проверено bot-10) | +0.019802, P 0.976, LOFO +0.023935, 5/5, new34 +0.008397 | ветка archive/worked-20261031: archive_review/docs_bots/2026-10-29/bot-10/REPORT10_accept.md §1
29.10 22:15 MSK | W4 | гейт FR1h, TOFR−TO (БЕЗ D-FINE R) | +0.004042, P 0.999, LOFO +0.004228, 5/5, new34 +0.00044 | ветка archive/worked-20261031: archive_review/docs_bots/2026-10-29/bot-10/REPORT11_audit.md C3h
29.10 22:15 MSK | W4 | ВНИМАНИЕ: при R строки sp_or/sp_focal/sp_focal2 вне 132 SpOr-участков = 0 ⇒ на трейне FR1h поверх R даёт 0; ожидание v3 ≈ +0.0198 к A, не +0.0239 | 0 | ветка archive/worked-20261031: archive_review/docs_bots/2026-10-29/bot-10/PREREG_aeor.md §1
29.10 22:15 MSK | W4 | перезамер v3 одним пакетом на коробке невозможен: нет table.pkl / OOF-стека A (F:/expds3/work на ПК, ПК офлайн) | — | orch/bot-W4/STATUS.md
29.10 22:15 MSK | W4 | v3 e2e на ПК (pc_e2e_v3.ps1) и замер времени 4090 — не сделаны | — | docs/STATUS_2026-10-29.md

01.11 19:47 MSK | bot7 | geo-gate S3+helper `11a47d2a` + пакет dmitrijkv/koz3-bot7-geo-gate READY (не пушен); suite 1016p/2 prefails; PR #26 draft | bots/bot7-geo-gate
## 29.10 22:55 MSK — W4: воспроизведение v2 (eb268d6, LB 0.41238) из main — ветка bots/w4-repro-v2 56cdf39

* Причина разрыва ≈−0.012: в main нет 4 release-артефактов v2 — `expds/geo_pixgbm.py`, `models/pixgbm/all/{poly,finds}.txt`,
  `models/rerank_arch_lgbm.txt` (сняты интеграцией 29.10 09:18–09:29). Без модуля pipeline молча уходит в «классику» геофизики.
* Фикс (56cdf39): `scripts/v2_release_artifacts.py` (build-time injection из git-объектов eb268d6, сверка sha256, `--check`),
  `scripts/golden_compare.py` (постадийные счётчики, первая расходящаяся стадия), `.gitignore`, `docs/REPRO_V2.md`. Main не тронут.
* main 70dedf4 + артефакты ≡ пересборка eb268d6: result.geojson побайтно равен на synth, 210, 214, 217, 044, 181 (CPU, threads=1).
  Остальной дрейф кода (16 файлов) на выход не влияет. Первая расходящаяся стадия у «main без артефактов» — geophysics.

| вход | стадия | ориг. (лог бота-7) | пересборка v2 | main+арт. | main классика |
|---|---|---|---|---|---|
| synth | geophysics / записано | +494 / 230 | +494 / 230 | +494 / 230 | +72 / 80 |
| 210 | geophysics / записано | +1542 / 383 | +1542 / 380 | +1542 / 380 | +369 / 369 |
| 214 | geophysics / записано | +64 / 56 | +64 / 56 | +64 / 56 | +37 / 125 |
| 217 | geophysics / записано | +722 / 298 | +722 / 298 | +722 / 298 | +113 / 170 |
| 044 | записано | — | 2 | 2 | — |
| 181 | записано | — | 486 | 486 | — |

* Остаток: 210 — 380 vs 383 (до порога fp1400/op142 vs 1399/143) одинаково у обеих сборок на коробке → окружение/копия входа, не код.
* Оригинальный zip (sha 8f3f2671…) только на ПК Елены — по sha не сверен (sha zip зависит от mtime); сверено по составу/CRC/размеру 525158901 Б.
* КОМПЛАЕНС: в main отслеживаются копии geo_pixgbm.py в analysis/…/bot-4/…/{pixgbm428,v2eb268d6}/expds/ и analysis/…/bot-7/work5/pc/tools/v2/expds/
  (найдено `git ls-files`, содержимое W4 не открывал).

* 22:55 MSK, bot 1's decision: 210 380 vs 383 — environment (py3.12.15 vs 3.12.16 / S3 copy of the input), not code; recorded in docs/REPRO_V2.md.
  The v2 line is only a comparison reference; the final archives contain no EN-derived files. The analysis/ copies of geo_pixgbm.py are not touched by W4.

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
| 29.10 | коробка /tmp/w4rp/in, in2 (вне git) | S3: участки 210/214/217 (tif+UTM.json, 36 ф.) и 044/181 (tif+las+UTM.json, 13 ф.) | 151,5 МБ + 129 МБ | W4 | после отчёта (rm -rf /tmp/w4rp/in*) |
