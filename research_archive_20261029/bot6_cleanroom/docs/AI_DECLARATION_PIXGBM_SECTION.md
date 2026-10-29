
## Дополнение PIXGBM (bot-6, 29.10.2026)

Существующие строки таблицы AI_DECLARATION не удаляются и не переписываются. Ниже — уточнение провенанса и инструментов bot-6.

### Провенанс модуля и моделей

* Модуль `expds/geo_pixgbm.py` написан ИИ-агентом **bot-3** (Cursor) по мотивам разбора публичного решения конкурента **EhimenNathan-EXpedition** (ветка `bot3-ehimen`, `expds/geo_finds_gbm.py`). В `main` не хранится; sha256 эталона сборки v2 — `35e4331bbc690e7eb3b31e07cc290837ce9fd96a05c92f4c999f24eceafda35b` (`docs/PIXGBM_DECISION.md`).
* Модели LightGBM `poly.txt` / `finds.txt` обучены bot-3 на трейне КОЗ-3 (регионы 200–217), `lightgbm==4.7.0`, `seed=0`, `ROUNDS=300`. Хеши — `models/pixgbm/all/MODELS_NOT_IN_GIT.md`.
* Решение владельца 29.10: временная декларация EN-линии + план **чистой переписи** отдельными ролями (спека / исполнитель без доступа к EN-коду). Код чистой переписи в этом дополнении **не пишется**.

### Запись в журнал (добавить строку в таблицу деклараций)

| Дата | Инструмент | Провайдер | Что сделано | Файлы / артефакты | Подпись |
|---|---|---|---|---|---|
| 29.10.2026 | Grok Bot (агент bot-6), Cursor | xAI / Cursor | Разбор PIXGBM: пути обучения, детерминизм, SIGTERM fork-пула, скрипт переобучения за флагом OFF, секции DATA_PROTOCOL / AI_DECLARATION; ветка `bots/bot6-pixgbm-cleanroom` | `scripts/pixgbm_retrain_cleanroom.py`, `tests/test_bot6_pix_pool_sigterm.py`, `docs/DATA_PROTOCOL.md` (§ PIXGBM), этот блок, `reports/bot6_pixgbm.md` | _ФИО_ |

### Детерминизм (заявление)

Повторный прогон `scripts/pixgbm.py train` с исходным `G.LGBM` (без `deterministic`/`force_col_wise`, `num_threads=2`, bagging) **не гарантирует** бит-в-бит совпадение текста модели. Скрипт bot-6 при `--enable-retrain` включает детерминистические параметры и сверяет sha256; при расхождении — допуск по `predict` на фиксированной матрице (`--metric-tol`).
