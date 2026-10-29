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
