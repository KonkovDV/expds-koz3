# Разбор новых материалов организаторов (30.10.2026, W0-docs)
Источники: `docs_org/Консультация №1, КОЗ №3 Экспедиция Data Science.pdf`, `docs_org/templates/{README.md,ai_declaration.md}`, `TR_KOZ3_approved.txt` (п. 8.5.3, 13.6, 13.10, Прил. 5). Сверка с `origin/main@70dedf4` и `bots/bot5-candidate-v4@ed73c5e`.

## Новые факты (добавлены в local_ctx/COMPETITION_RULES.md)
1. Финал — **17–19.11**, лимит попыток на весь период; без загрузки авто-берутся 2 лучшие публичные посылки. Допуск: max публичный скор > 0.
2. Образ — публичный DockerHub; шаблон: `docs/README.md`, `docs/ai_declaration.md`, `inference/main.py` — единственная точка входа.
3. Метрика: частичное перекрытие или центр; точки — по расстоянию; контур не влияет; FP ×0,5.
4. Размеры классов геофизики (MagGr obj 1–30, struct 2–400; EtGr 0,3–12, >2 м — геология; TerFl obj 0,3–10, struct 1–50). Папки EtGr: `08_*_карты` и `08_*_профили`. Георадар заявлен, в данных нет.

## Расхождения и риски
| # | Где | Что | Риск | Действие |
|---|---|---|---|---|
| R1 | `bots/bot5-candidate-v4`: `expds/geo_pixgbm.py`, `configs/default.toml:295` `[geophys.pixgbm] enabled` | модуль «по мотивам» решения EhimenNathan в пути архива (+ `models/pixgbm/all/*`) | **высокий** (clean-room, экспертиза 13.6/8.5) | QUEUE DOC-1 (W5): выключить или заменить clean-room/GEO-ORG1 |
| R2 | `origin/main`: `configs/default.toml:295` pixgbm enabled, модуля нет → `expds/pipeline.py:361` тихий откат | флаг расходится с кодом | средний | DOC-1: явно `enabled=false` |
| R3 | `orch/QUEUE.json` FINAL 11.11 | по консультации финал 17–19.11 | средний (план) | DOC-3: решение Елены |
| R4 | `scripts/build_submission.py:38-80` | `docs/` в архив платформы не входят | низкий: для платформы не нужно, на экспертизу — весь репо | `docs/README_FINAL.md` на ветке `bots/w0-docs` |
| R5 | `expds/geophys.py:557-558` AREA_M2 | пределы шире описания; MagGr obj от 5 м² > 1 м² | низкий (FP ×0,5) | DOC-4 (W4) |
| R6 | `inference/main.py:21` OMP_NUM_THREADS=8 при 24 ядрах | недогруз CPU геофизики | низкий (время 3–4 мин из 30) | нет |
| ✓ | `expds/constants.py:67-69`, `expds/writer.py:133-163` | 20 000 объектов, 3000 вершин, 100 МБ, только Polygon | ок | — |
| ✓ | `metadata.json`, `docker/Dockerfile` | образ публичный без кода/весов, entry `python inference/main.py` | ок | — |
| ✓ | `expds/catalog.py:156`, `expds/naming.py:81` | георадар пропускается; профили EtGr распознаются | ок | — |

## Документы
Ветка `bots/w0-docs`: `docs/README_FINAL.md` (по шаблону), `docs/ai_declaration.md` (+ Grok Bot 1–5, §2а код организаторов и веса с лицензиями). Шаблон кладёт их в `docs/`; перед FREEZE — DOC-2.
