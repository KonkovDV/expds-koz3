# STATUS (KOZ-3) — формат: время MSK | бот | что | число | ссылка

29.10 08:47 MSK | W4 | лучший официальный результат: v2 eb268d6 | 0.41238 | docs/LB_LOG.md
29.10 22:15 MSK | W4 | ветка bots/w4-gate от bots/bot7-candidate-v3fix-nomodels 0037c0e (v3 + CRS-фикс 008_Герите) | — | bots/w4-gate
29.10 22:15 MSK | W4 | v3 за флагами: в configs/default.toml sp_dfine=false и [post.frame_prior] закомментирован (=выкл); включённая сборка — configs/v3_on.toml (= bot-7 work8/default_v3on.toml) через build_submission_git.py --config | — | configs/v3_on.toml
29.10 22:15 MSK | W4 | гейт D-FINE R (SpOr, f0) против A, honest-10 fam10 (bot-9 paired_5f, проверено bot-10) | +0.019802, P 0.976, LOFO +0.023935, 5/5, new34 +0.008397 | docs/bots/2026-10-29/bot-10/REPORT10_accept.md §1
29.10 22:15 MSK | W4 | гейт FR1h, TOFR−TO (БЕЗ D-FINE R) | +0.004042, P 0.999, LOFO +0.004228, 5/5, new34 +0.00044 | docs/bots/2026-10-29/bot-10/REPORT11_audit.md C3h
29.10 22:15 MSK | W4 | ВНИМАНИЕ: при R строки sp_or/sp_focal/sp_focal2 вне 132 SpOr-участков = 0 ⇒ на трейне FR1h поверх R даёт 0; ожидание v3 ≈ +0.0198 к A, не +0.0239 | 0 | docs/bots/2026-10-29/bot-10/PREREG_aeor.md §1
29.10 22:15 MSK | W4 | перезамер v3 одним пакетом на коробке невозможен: нет table.pkl / OOF-стека A (F:/expds3/work на ПК, ПК офлайн) | — | orch/bot-W4/STATUS.md
29.10 22:15 MSK | W4 | v3 e2e на ПК (pc_e2e_v3.ps1) и замер времени 4090 — не сделаны | — | docs/STATUS_2026-10-29.md

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
| 29.10 | коробка /tmp/w4rp/in, in2 (вне git) | S3: участки 210/214/217 (tif+UTM.json, 36 ф.) и 044/181 (tif+las+UTM.json, 13 ф.) | 151,5 МБ + 129 МБ | W4 | после отчёта (rm -rf /tmp/w4rp/in*) |
