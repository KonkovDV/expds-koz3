# Обзор репозитория KonkovDV/expds-koz3 после 29.10 18:00 MSK

**Дата отчёта:** 01.11.2026 (Europe/Moscow, UTC+3)  
**Срез:** read-only; клон `/tmp/arch` после `git fetch origin`; remotes `origin/*`.  
**Кончик main:** `f127db8` (31.10 23:54 MSK, author expds-local) — «Let the overnight run fine-tune the shipped center map instead of ImageNet.»  
**База площадки:** **0,41238** = v2 `eb268d6` (`submission_bot7cand_v2_eb268d6.zip`).  
**Не база:** v3a_min **0,37838** (30.10) — геофизика срезана.  
**Дедлайн:** 11.11.2026.  
**pytest на копии main** (`/tmp/koz3_main_wt` @ `f127db8`, `/workspace/koz3-venv`): **999 passed, 64 skipped** за 134 с (≈2:14). Репо не менялось, push/Kaggle/PC/сообщения ботам не делались.

Архивный план сравнения: ветка `archive/research-results-20261029` → `research_archive_20261029/PLAN_2026-10-29.md` (P1–P10).  
Актуальное «держать 0.41238»: `docs/plans/PLAN_TO_1111_20261031.md` на main.

---

## 1. main: коммиты, ключевые файлы, путь сдачи vs v2

### 1.1 Журнал (после 29.10 18:00)

На `origin/main` с cutoff ~423 коммита (много мелких orch/ботов и merge). Ключевые вехи:

| Когда (MSK) | SHA / событие | Суть |
|---|---|---|
| 30.10 ~23:54 | merge `7275f1d` «Replace the broken automatic merge with the repaired tree» | Починенное дерево; в коммите заявлено **957 passed / 70 skipped** |
| 31.10 утро–день | STATUS / LB_LOG / планы | Аудит: база 0.41238; v3a_min 0.37838; FR1h PASS; r50c5 REJECT |
| 31.10 вечер | пилоты dino_gor / near_kept / peak-rerank / WSR код | См. STATUS + `orch/COORD_LOG.md` на `bots/orch-protocol` |
| 31.10 23:54 | tip `f127db8` | Промт ночи: fine-tune с `--init-weights models/spor_focal_r18.pt` (не ImageNet) |

Авторы tip-коммитов main после cutoff (по `git log`): в основном `expds-local`, также bot-8, bot5-w4, bot-4, coord-lease, bot-W2, Konkov Dmitrij, bot-W3/W4/W5, grok-bot и др.

### 1.2 Документы на tip

| Файл | Состояние |
|---|---|
| `docs/STATUS.md` | Актуальный журнал до **31.10 23:55** (ночь WSR как план). **Нет** записи STOP WSR 01.11 00:06 — она только в `orch/COORD_LOG.md` на ветке `bots/orch-protocol` |
| `docs/LB_LOG.md` | База 0.41238; v3a_min 0.37838 в прозе; **в markdown-таблице посылок строки v3a_min нет** |
| `docs/plans/*` | `PLAN_TO_1111_20261031.md`, `PLAN_SP_LARGE_ONE_GPU_20261031.md`, `PROMPT_DINOV3_HILLFORT_KAGGLE_20261031.md`, `PROMPT_WITHIN_SITE_RANK_20261031.md` |
| `README.md` | База 0.41238; v3 zip ждёт e2e на ПК; ссылки на archive |
| `CHANGELOG.md` | Срез в основном 29.10 (выгрузки, TR, harvest, li_decode) |
| `FOR_BOT9_GORLOC.md`, `FOR_BOT9_DINOV2REG.md` | Инструкции/ячейки для bot-9; dino_gor линия позже STOP |
| `HANDOFF_W2.md` | Полный handoff W2 (r50c5, DEM fill FAIL, аренды, датасеты) — r50c5 уже REJECT на honest-10 |
| `orch/*` | На tip main есть каталог orch; **актуальный COORD_LOG/leases** смотреть на `bots/orch-protocol` @ `a4856de` (01.11 00:08) |

### 1.3 `configs/default.toml`: ON/OFF vs `eb268d6`

Секции с `enabled` (сравнение tip main ↔ `eb268d6`):

| Секция | eb268d6 | main tip | Комментарий |
|---|---|---|---|
| `[geophys.pixgbm]` | **true** | **false** | Главный разрыв пути сдачи |
| `[rerank_arch]` | **true** | **false** | `models/rerank_arch_lgbm.txt` **нет в git** tip |
| `[hillfort]`, `[verify]` | true | true | Без изменений флага |
| `[ae_recall_relax]`, `[aeor_dfine]`, `[dfine_ae_tile]`, `[dfine_aeor]`, `[gor_dense]`, `[gor_gate]`, `[large_objects]`, `[li_cnx]`, `[li_half]`, `[orgate]`, `[sat_thr_relax]`, `[sp_dfine]`, `[speed]` | нет / — | **false** | Новые флаги, default OFF |
| `[post.frame_prior]` (FR1h) | — | **закомментирован** | PASS honest-10; в default не включать без «да» |

Дополнительно на tip: `org_maggr` / `org_etgr` / GEO-ORG флаги **false**; `dome` / `slope_gate` false (по STATUS).

### 1.4 Что именно изменилось в пути сдачи vs v2 `eb268d6`

1. **Нет `expds/geo_pixgbm.py` на tip** (на `eb268d6` есть). Pipeline без модуля уходит в «классику» геофизики → на синтетике/участках резко меньше объектов гео (таблица в STATUS: 210: +1542→+369 и т.п.). Это объясняет падение LB v3a_min **0.37838** (гео 0.769/0.656 в архиве сломаны).
2. **`[geophys.pixgbm] enabled = false`**, нет `models/pixgbm/all/{poly,finds}.txt` на tip (были у v2).
3. **`[rerank_arch] enabled = false`**, нет `models/rerank_arch_lgbm.txt` (есть только `.json`).
4. Много **нового кода за флагами OFF**: D-FINE SpOr/Ae/Or, large_objects, sat_thr_relax, orsp/tt a, near_kept, hillfort_loc, within_site_rank hinge, CNX lidar, speed, orgate, GEO-ORG1–3 и т.д.
5. Добавлены `inference/cnx/` (vendor RVT + so), раздут `pipeline.py` / `post.py`.
6. Чистый клон tip **не** воспроизводит LB 0.41238 без инъекции артефактов v2 (`docs/REPRO_V2.md`, ветка `bots/w4-repro-v2`): разрыв ≈ −0.012 на геофизике.

**Вывод:** tip main — «безопасный» каркас с выключенным PIXGBM/rerank; действующая база площадки — **внешний** zip v2, не то, что соберёт `build_submission` с tip без ручной подкладки артефактов.

### 1.5 Новые каталоги vs `eb268d6`

Появились (неполный список): `orch/`, `orch_bot-W2/`, `kaggle/`, `kernels/`, `centerhead/`, `pc/`, `reports/`, `results/`, `tools/`, серия `w2_*` (`w2_datasets`, `w2_demfill`, `w2_gate`, `w2_gaudit`, `w2_kernels`, `w2_merge`, `w2_notes`, `w2_p11`, `w2_tools`), `w3/`, плюс разросшиеся `docs/plans`, `patches`, `analysis` (на tip — в основном тесты PIXGBM, не сам модуль).

Ветка `archive/worked-20261031` (@ `2278ff2`): закрытые отчёты / «frozen geophysics backup» — **не** tip сдачи.

---

## 2. Ветки с коммитами после 29.10 18:00

| Ветка | Tip (MSK) | SHA | Что делает | Замер | Вердикт |
|---|---|---|---|---|---|
| `origin/main` (= `origin`) | 31.10 23:54 | `f127db8` | Каркас + планы + код пилотов; PIXGBM/rerank OFF | pytest 999p/64s (этот обзор); honest-10 отдельных флагов — в STATUS | База кода; LB tip ≠ 0.41238 |
| `bots/orch-protocol` | 01.11 00:08 | `a4856de` | Аренды, QUEUE, COORD_LOG, протокол | Журнал пилотов | Источник истины по ночным STOP |
| `bots/a-within-site-rank` | 31.10 23:59 | `d25ca53` | Ядро `elenakonkova/koz3-wsr-f0` (nn_oof + WSR) | Pilot fold0 01.11 00:06 | **STOP** (median FP 9→16; damaged F1 0.6636→0.5561) |
| `bots/w2-lidar-aug` | 01.11 11:55 | `e514195` | Lidar relief/zoom aug + site-balanced; kernel `koz3-lidar-aug-f0` | На tip — push ядра; **итог COMPLETE/гейт в STATUS/COORD_LOG tip не зафиксирован** | **Неясно** (нужен статус ядра / gate JSON) |
| `bots/bot2-e2e-v3-prep` | 01.11 10:36* | `fadf8e3` | `drop_groups_covered`: не снимать v2, если D-FINE не закрыл | Код; honest-10 **не** снят на ветке | Открыто |
| `bots/bot3-aerial-recall` | 01.11 ~10:33* | `7b1c272` | Default-off aerial recall + honest-10 scorer | OOF/honest-10 **нет** на машине автора (по сообщению) | Открыто / нет замера |
| `bots/bot4-large-objects` | 01.11 ~10:44* | `33791bc` | CPU runner large-object; OOF cache отсутствует | Нет OOF | Открыто |
| `bots/bot5-candidate-v4` | 01.11 ~10:47* | `0484819` | v4: SpOr D-FINE выкл; pytest claim ~1004p | Локальные тесты; не LB | Кандидат-каркас |
| `bots/bot6-pixgbm-cleanroom` | 01.11 ~10:45* | `9b411c4` | Clean-room PIXGBM, pin retrain, SIGTERM repro | Документация/репро; не LB | В работе |
| `bots/bot8-orsp-tta` | 01.11 ~10:45* | `f3aafa4` | `[orsp_tta]` OFF; TTA sat U-Net; D-FINE views unmeasured | Нет honest-10 | Открыто |
| `archive/worked-20261031` | 31.10 20:46 | `2278ff2` | Архив закрытых отчётов + geo backup | Исторические гейты | Архив |
| `archive/research-results-20261029` | (до cutoff) | — | План P1–P10, BOARD/BACKLOG/LESSONS | — | База сравнения плана |
| `master` | 28.10 | `24bd916` | Предок main (845 commits behind) | — | Устарел |

\* Время tip bot2–8 в логе — UTC+0 в части записей Cursor Agent; по смыслу утро 01.11 MSK.  
Много старых `bots/*` из BRANCHES.md 29.10 в этом срезе tip не обновлялись после cutoff — не перечисляются как «новые».

Дополнительно из COORD_LOG (локальные/короткоживущие ветки A): `bots/a-dino-gor`, `bots/a-peak-rerank` — пилоты **STOP**, default.toml не трогали.

---

## 3. LB и таблица замеров

### 3.1 Посылки площадки (из `docs/LB_LOG.md`)

| Дата | Архив / коммит | Балл | Примечание |
|---|---|---|---|
| 23.10 23:36 | — | 0,06140 | Только гео |
| 24.10 08:35 | — | 0,32850 | |
| 24.10 13:37 | — | 0,32850 | |
| 27.10 19:17 | hillfort | 0,33352 | |
| 27.10 23:27 | `submission_arch.zip` | 0,34796 | |
| 28.10 08:13 | нет данных | 0,34757 | В шуме |
| **29.10 08:47** | **`submission_bot7cand_v2_eb268d6.zip` / `eb268d6`** | **0,41238** | **Действующая база** |
| **30.10** | **v3a_min** | **0,37838** | В прозе LB_LOG/STATUS; **нет строки в таблице**; не загружать снова |

Официальный разбор v2 (`orch/OFFICIAL_SCORES.md` / лог): soft recall 0,622; geo P/R **0,769/0,656**; satellite recall **0,255**; aerial **0,013**; ortho 0,140; корзина 5–10k м² recall **0**.

### 3.2 Честные замеры / пилоты (после 29.10 вечера → 01.11)

| Что | Число | Вердикт | Где |
|---|---|---|---|
| Base A (honest-10) | ≈ **0,314615** | опора гейтов | STATUS / HANDOFF_W2 |
| FR1h `[post.frame_prior]` | Δ **+0,00409**, P **0,999**, damaged **+0,0107** | **PASS** (не в default без «да») | STATUS 31.10 15:24; `out/fr1h-gate/` |
| FR1h (ранее W4/bot-10, без R) | +0,004042, P 0,999 | PASS | archive bot-10 REPORT11 |
| D-FINE R SpOr (историч.) | +0,019802, P 0,976 | PASS vs A (ожидание v3) | archive REPORT10 |
| r50c5 lidar varb | Δ **−0,00513**, P 0,33 | **REJECT** | STATUS; PLAN_TO_1111 |
| r50c5 varb_kur | Δ +0,00088, P 0,56 | **REJECT** | то же |
| r50c5 early-stop f01 | mean Li 0,4395 | PASS early ≠ submit | STATUS |
| sp_large fold0 | F1 dmg 0,3537 < base 0,3677; hits↑ но FP↑ | **STOP** | COORD_LOG 31.10 17:12 |
| near_kept / rescue ring | Δ **−0,00188**, P 0,245 | **FAIL** | STATUS §10 |
| sp_dino_gor fold0 | hits 3, P 0,33, F1 0,182 | numeric PASS впритык → линия **STOP** | STATUS / COORD_LOG |
| peak-rerank pilot | lifted 25, P 0,20, med miss 851 м, AUC 0,973 | **STOP** все 4 гейта | STATUS / COORD_LOG |
| within-site-rank (WSR) f0 | median FP 9→16; dmg F1 0,6636→0,5561 | **STOP** 01.11 00:06 | **только** COORD_LOG orch-protocol |
| GEO-ORG1 alone | ниже порога | FAIL | STATUS W4 |
| C-et+ORG3 / v2-clean пакеты | Δ ~+0,02 vs A18 | PASS прокси; zip **не** на LB | STATUS 30.10 |
| DEM fill P-W2-10 | FAIL | закрыто | HANDOFF_W2 |
| Диски 5 см (аудит) | ~+0,004 (оценка) | вне посылки; не в default | STATUS 31.10 23:00 |

---

## 4. Кто чем занят; планы на ночь/день

| Кто | Ветка / артефакт | Роль |
|---|---|---|
| **Elena / владелец** | решения «да» на LB; лицензии моделей (31.10 21:16) | Гейт политики |
| **Konkov Dmitrij** | push в main/ветки | Владелец репо |
| **expds-local** | основной поток коммитов main | Сборка tip, промты |
| **coord / bot A / koz3-A** | `bots/orch-protocol`, пилоты dino/peak/WSR, `bots/w2-lidar-aug` | Оркестрация Kaggle, аренды |
| **W2 / bot-W2-lidar** | `HANDOFF_W2.md`, w2_*, r50c5 (закрыт REJECT) | Лидар CNX |
| **bot-2** | `bots/bot2-e2e-v3-prep` | Covered-drop / sat prep |
| **bot-3** | `bots/bot3-aerial-recall` | Aerial recall флаги |
| **bot-4** | `bots/bot4-large-objects` | Крупные объекты |
| **bot-5 / bot5-w4** | `bots/bot5-candidate-v4`, v2-clean | Кандидаты / geo clean |
| **bot-6** | `bots/bot6-pixgbm-cleanroom` | Clean-room PIXGBM |
| **bot-8** | `bots/bot8-orsp-tta` + много коммитов main | TTA, сторожа, метрика |
| **W4 gate** | STATUS geo ORG*, REPRO_V2 | Гейты / воспроизведение v2 |
| **bot-9** | `FOR_BOT9_*.md` | Ядра gorloc / dinov2reg (линия gor → STOP) |
| **W5** | (историч. rerank) | Rerank поток |
| **Cursor Agent** | утренние tip bot2–8 01.11 | Документация/флаги без замеров |

**Ночь 31.10→01.11 (по COORD_LOG):** один GPU `elenakonkova/koz3-wsr-f0` → **STOP 00:06**. До этого за вечер: dino_gor STOP, peak-rerank STOP, near_kept FAIL. sp_large ранее STOP.  
**День 01.11:** tip `bots/w2-lidar-aug` — ядро `koz3-lidar-aug-f0` (результат гейта **неясен** на момент обзора).  
**Планы в промтах:** WSR (уже STOP), DINOv3 hillfort (STOP), sp_large (STOP), PLAN_TO_1111 — держать 0.41238, бить спутник только с OOF+тройным гейтом.

---

## 5. Сравнение с планом 29.10 (P1–P10)

Источник: `archive/research-results-20261029` → `PLAN_2026-10-29.md`.

| ID | Суть | Статус на 01.11 | Комментарий |
|---|---|---|---|
| **P1** | Aerial D-FINE Ae + ae_fort relax | **Открыто / устарело по приоритету** | Код на bot3; честный e2e/гейт на ПК по STATUS не закрыт; PLAN_TO_1111 не ставит aerial первым |
| **P2** | Aerial tiling+NMS+budget | **Открыто** | Зависит от P1 |
| **P3** | Sat thr relax U-Net 0.85 | **Открыто, флаг OFF** | `sat_thr_relax=false`; без отдельного гейта не включать (PLAN_SP_LARGE) |
| **P4** | Разбор сдвигов 10–50 м CRS | **Неясно / частично** | Явного закрывающего отчёта на tip STATUS нет |
| **P5** | Крупные объекты / окно 320 | **Открыто; sp_large STOP** | bot4 без OOF; GPU sp_large fold0 STOP |
| **P6** | Ortho Ae→Or | **Открыто / отложено** | GPU ушёл в r50c5/sp_large/dino/WSR |
| **P7** | TTA/multiscale | **Код есть, замер нет** | `bots/bot8-orsp-tta`, enabled false |
| **P8** | c3-sc lidar LOFO+seeds | **Закрыто отрицательно как r50c5** | REJECT honest-10 |
| **P9** | Честная калибровка порогов | **Частично** | Инфра гейтов жива; массовый honest-10 кандидатов P1–P5 не закрыт |
| **P10** | Офлайн-скорер как LB | **Неясно** | Инструменты metric/bot8 есть; «как LB» доведение не подтверждено tip |

**Устарело относительно 29.10:**
- Совокупный прогноз 0.45–0.51 и цель 0.61 — `PLAN_TO_1111` прямо говорит: **0.20 не набрать честно за оставшиеся дни**; ближайшее — **не опуститься ниже 0.41238**.
- Ожидание «следующий замер = r50c5» (STATUS утро 31.10) — **уже REJECT**.
- Запрет DINOv3 по лицензии в старых текстах — **снят** владельцем 31.10; но пилоты DINOv3 **STOP** по качеству.
- tip main **без** PIXGBM ≠ план «улучшать поверх v2».

**Сделано / закрыто иначе:** FR1h PASS; починка merge-дерева; orch leases; аудит LB drop; REPRO_V2; ряд отрицательных пилотов задокументированы.

---

## 6. Риски и несогласованности

1. **Путь сдачи tip ≠ LB 0.41238** — PIXGBM/rerank выключены/файлы сняты; риск случайного zip «как main» → снова ~0.38.
2. **LB_LOG:** v3a_min есть в тексте, **нет в таблице** посылок.
3. **STATUS tip устарел относительно orch:** WSR STOP 01.11 00:06 не попал в `docs/STATUS.md`.
4. **Дублирование orch:** актуальный протокол на `bots/orch-protocol`; на main тоже есть `orch/` — риск рассинхрона.
5. **check_conflicts** (COORD_LOG): ERROR≈5–6 — грязный общий клон, worktree вне `wt/`, TPU smoke без аренды.
6. **Правило «одна ветка — один владелец»** нарушалось 29.10 (bot-8 в чужие ветки) — зафиксировано в BRANCHES.md.
7. **Пуши в main** шли плотным потоком (expds-local и боты) — формально против старого «не в main без да», но это принятый рабочий режим tip; риск шума vs archive.
8. **v3a_min на LB без удержания geo 0.769/0.656** — нарушение духа «не ломать гео»; урок записан.
9. **Секреты:** в tip **не найдено** отслеживаемых `cred.json` / `kaggle.json` / `.env` / pem. В HANDOFF/планах — инструкции брать ключи из env/датасетов, **не** коммитить. Паттерны загрузки cred в коде есть — это не утечка значений. **Не проверялись** бинарные/история на вшитые ключи полностью (ограничение обзора).
10. **Лимит 100 МБ:** по TR — на **result.geojson** / валидатор (writer budget ~97 МБ). Дерево tip ≈ **873 МБ** tracked (много `.pt` ~55 МБ) — это вес репо/образа, не лимит одной посылки-результата. Zip v2 ~525 МБ на ПК (STATUS) — образ инференса.
11. **pytest:** на этом срезе **999 passed / 64 skipped**, падений нет. (Merge 30.10 заявлял 957/70 — число тестов выросло.)
12. **analysis/ копии geo_pixgbm:** на tip main модуля нет; STATUS W4 раньше упоминал копии в analysis — сейчас `git ls-tree` tip даёт только `tests/test_geo_pixgbm_*.py`. Уточнять на archive при сборке.
13. **Неясно:** финальный статус ядра `koz3-lidar-aug-f0` после 01.11 11:55; жив ли OOF-стек A на боксе/ПК для гейтов bot3/4/8; sha256 загруженного v2 zip на площадке по-прежнему «нет в логе».

---

## 7. Наиболее перспективные следующие шаги (к LB > 0.41238)

Только то, что опирается на уже измеренное или явно разрешённое; без выдуманных приростов.

1. **Сборка «v2-hold» = артефакты PIXGBM+rerank v2 + опционально FR1h (после thr re-sweep и проверки geo 0.769/0.656)** — единственный PASS ~+0.004 с тройным гейтом, ещё не в default. Без «да» Elena на LB не класть; сначала Docker/mock_val.
2. **Спутниковая дыра (recall 0.255)** — не новые экзотические головы (dino/peak/WSR/sp_large уже STOP), а **существующий** флаг/голова с **полным honest-10** (Δ≥0.003, P≥0.9, damaged не хуже) + неизменная гео. Кандидаты к гейту: bot2 covered-drop + аккуратный sat путь; не `sat_thr_relax` вслепую.
3. **Aerial D-FINE (P1) с OOF** — исторический гейт +0.0198 vs A; на LB не проверялся поверх v2; нужен e2e на ПК и проверка, что geo не падает. Код на bot3 / флаги OFF.
4. **Clean-room PIXGBM (bot6)** — вернуть geo-уровень архива **легально** (DOC-1), чтобы tip снова мог собирать ≥0.41238 без ручной инъекции из `eb268d6`.
5. **Узнать статус `koz3-lidar-aug-f0`** — если PASS early/gate, тогда honest-10; если нет — сразу в журнал отказов, GPU не жечь.

Антиприоритет: повторы r50c5, near_kept, dino_gor, peak-rerank, WSR, sp_large как были; LB без замера; zip с tip без PIXGBM.

---

## 8. Ссылки (быстрый указатель)

- Tip: `origin/main` @ `f127db8`
- База LB: `eb268d6`, `docs/LB_LOG.md`, `docs/background/LB_SCORING_v2_eb268d6_0.41238.log`
- Статус: `docs/STATUS.md` (отстаёт от WSR STOP)
- Оркестрация: `bots/orch-protocol` @ `a4856de` → `orch/COORD_LOG.md`
- План 29.10: `archive/research-results-20261029` / `research_archive_20261029/PLAN_2026-10-29.md`
- План до 11.11: `docs/plans/PLAN_TO_1111_20261031.md`
- REPRO v2: `docs/REPRO_V2.md` / ветка `bots/w4-repro-v2`
- Архив отчётов: `archive/worked-20261031`

*Конец отчёта. Ничего не выдумано; пробелы помечены «неясно».*
