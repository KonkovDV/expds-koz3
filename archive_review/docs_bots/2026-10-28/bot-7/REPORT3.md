# bot-7 · REPORT3 · кандидат на посылку, финальный состав (28.10.2026, МСК; обновлено 22:45 МСК)

## Вывод (3 строки)
1. Ветка `bot7-candidate` (HEAD `4c3e29e`, дерево `534e382f`, локально, без push). Состав: stack6 separate + курганы 0,5 м + реранк архитектуры + crop verify (fix bot-8) + надёжность bot-8 (`2dd45fa 2621e33 7a9f1d1 fd7dd50`) + ae_fort (уже в базе); `kurgan_twins_only` bot-6 взят **выключенным**. Микрокругов архитектуры нет.
2. Лимиты: на прокси всех 103 семей ≤ 3000 вершин и ≤ 20000 объектов держатся, `validate` OK. Добавлен последний рубеж `writer.enforce_limits`. pytest 327 passed / 9 skipped. `check_submission` PASS на синтетике (52,9 с, 979 МиБ) и на реальном регионе (10,2 с, 313 МиБ). **ОЗУ ≤ 2 ГБ на полном регионе не проверено** (полный инференс на коробке запрещён).
3. Честный замер на коробке, без crop, unseen keep: stack6+kur **+0,016428** (P 0,994, LOFO +0,015121, 5/5). Twins поверх этого **+0,000352** (P 0,973, LOFO +0,000192, 4/5). Пакет с crop мерится только на ПК: две команды в §6, читают только C:. На LB ничего не посылалось.

## 1. Источники (цитаты)
- `bot-10/REPORT4.md` (21:17). stack6 sep: keep +0,015057, P 0,948 (A keep 0,333237), drop +0,006703, site +0,014454 (`bot-10/work4/stack6_global.json`); unified +0,0337 — артефакт. Rerank strict: +0,009536, P 0,957, LOFO +0,007869, 5/5, site +0,00959; с поправкой на отбор около +0,0035. kur: +0,001371, P 0,953. R+kur: +0,010907, P 0,979, CI [+0,00181; +0,01663], LOFO +0,007933. archM отвергнут. ae_fort +0,005240, P 0,926. Crop и kur не аддитивны, поэтому пакет мерить на ПК.
- `EXPERIMENTS.md`, «Сводка решений» (21:15). crop verify: PENDING-on-PC, +0,00414, P 0,984 (`/workspace/rerank/verify_sp_arch_board_2026-10-28.json`). Пакет: PENDING-on-PC, заготовка `<база> <флаги crop> <bot7-bundle только kur 0,5 м> --honest 10 --rerank stack_rerank_arch.geojson`. kurgan_twins (полный, bot-6/TYPES.md §5) +0,00285, bot-10 не перепроверял.
- `bot-8/HANDOFF_BOT7.md` — надёжность для кандидата.

## 2. Ветка и коммиты (`bot-7/repo`, база `24bd916`)
| коммит | что |
|---|---|
| `42ada8d` `dd7982e` | `post.micro_circles` с радиусом по классу; в конфиге только курганы 0,5 м |
| `07bed29` | bot-8 `8343483`: `[verify] weights` от корня архива |
| `733d3ed` `fe0e570` | `[rerank_arch]` (код, тесты) и `enabled = true`, модель `models/rerank_arch_lgbm.{txt,json}` |
| `2dd45fa` `2621e33` `7a9f1d1` `fd7dd50` | **bot-8 reliability** (= `bot8-reliability-on-bot7-candidate`; `git diff FETCH_HEAD fd7dd50` пуст, дублей нет). try вокруг геофизики, `safe_finish`, без NaN, `build_submission_git`/`check_submission`, дедлайн окна `[time] window_deadline = true, window_reserve_s = 30` |
| `c71af0b` | `[post] multipole_object_poly = true`, `filter_finds_points = true` (stack6 separate); тест состава проверяет и ae_fort в базе (`[models] ae`, порог ae.fortifikatsii 0,998215) |
| `f4d0159` | **новый guard** `writer.enforce_limits`: только Polygon, ≤ 3000 точек кольца, ≤ 20000 объектов, лог при срабатывании; `tests/test_bot7_limits.py` |
| `276f949` | stack6-шаги и в запасной цепочке `safe_finish`, каждый в своём try |
| `1d8fc71` | cherry-pick bot-6 `ad8ae9b` (`[post] kurgan_twins_only`, `SP_TWIN_GROUPS = (sp_focal, sp_or)`). Отдельно от `geom_opt.dedup_anchors`, их не связывал |
| `4c3e29e` | twins_only в `safe_finish`, как в `finish()`; в конфиге только комментарий (ВЫКЛ); тест «по умолчанию выкл. и в fallback» |

Плоский diff конфига `24bd916 → bot7-candidate` (`work3/toml_diff.txt`), 9 ключей:
```
post.filter_finds_points : <нет> -> True
post.micro_circles.radius_m.kurgany_povrezhdennye : <нет> -> 0.5
post.micro_circles.radius_m.kurgany_tselye : <нет> -> 0.5
post.multipole_object_poly : <нет> -> True
rerank_arch.enabled / meta / model : <нет> -> True / models/rerank_arch_lgbm.json / .txt
time.window_deadline : <нет> -> True
time.window_reserve_s : <нет> -> 30
```
Как включить twins: в `[post]` дописать `kurgan_twins_only = true`. Решение принимается после перемера на ПК вместе с пользователем.

**Порядок `safe_finish` (по запросу bot-8) проверен** по `expds/watchdog.py`. Реальный путь `checkpoint → safe_finish → finish` (`watchdog.py:216`, SIGTERM `:259`). `finish()` (`:65-112`): only_known_classes → rerank_arch → apply_thresholds → + reranked → merge_within_markup → multipole → filter_finds_points → kurgan_twins | twins_only → micro_circles (курганы 0,5 м) → finalize → cap_features. Запасная цепочка (`:115-170`) идёт в том же порядке, каждый шаг в своём try: rerank → пороги → + reranked → multipole → filter → twins_only → micro_circles → finalize по одному → cap. Сознательные отличия: нет merge (идея bot-8: «без склейки»), нет полного `kurgan_twins`, fill_holes, core, split и ellipse. В кандидате они все выключены. Порядок верный, правок не потребовалось, кроме моих `276f949` и `4c3e29e`.

## 3. Лимиты и время (коробка, 2 потока, nice 10, `ulimit -v 3500000`)
| проверка | результат | файл |
|---|---|---|
| прокси, каждая из 103 семей (1 386 709 строк стека до порогов): `watchdog.finish` + `writer.write_geojson` + validate | все OK; макс. вершин 2999; макс. объектов на семью 6522 (Романово); файл ≤ 17,6 МБ; finish+write ≤ 10,44 с (Гнёздово), сумма 67,1 с; RSS ≤ 466 МБ | `work4/limits_summary.json`, `limits_per_family.json` |
| крупнейшая семья (Вышегжа, 180 433 строки) | 2666 объектов, 4,6 с | там же |
| 6 крупнейших семей одним файлом (715 883 строки, 24 региона) | 9743 объекта, вершин ≤ 2997, 20,3 МБ, 25,7 с; **RSS 2291 МБ > 2 ГБ**. Это артефакт прокси: 716 тыс. shapely-объектов в одном процессе, реальный регион в разы меньше. Учитывать как риск | `limits_summary.json` |
| худший случай: ответы всех 103 семей в одном файле | 54 016 → 20 000 (`cap_features` срезал только fortifikatsii 38 742 → 4 726); 44,4 МБ, вершин ≤ 2999, validate OK; finish 137,3 с, cap+write 10,3 с, RSS 698 МБ | `work4/cap_all.json` |
| порядок stack6: «stack6 на сыром, потом finish» (bot-10) против «finish со stack6» (кандидат) | 0 различий на Гнёздово (3539), Семибратнее (4504), Судьбищи (205) | `work4/stack6_order.json` |
| pytest всего репо (`-k "not coco_roundtrip"`, нет cv2) | **327 passed, 9 skipped, 3 deselected** (218,7 с) | `work3/pytest_repo.log` |
| сборка `scripts/build_submission.py` | `work3/dist/submission_bot7cand_4c3e29e.zip`, 47 файлов, 523 218 857 Б, sha256 `7533ff26…962e` | `work3/build.log` |
| `check_submission` + смоук на синтетике (unshare, taskset 0,1, 2 потока, лимит 2000 МиБ) | **PASS, 27 проверок, FAIL 0, WARN 0**; main.py 52,9 с, пик 979 МиБ; 80 полигонов | `work3/check.log` |
| то же, вход — реальный регион `210_Гнёздово_Терраса_2` (только MagGr) | **PASS**; 10,2 с, 313 МиБ, 323 finds_points | `work3/check_real.log` |

**Что не проверено:** полный main.py на крупном реальном регионе с LAS и снимками (запрещено «Никаких тяжёлых инференсов»), а значит и ОЗУ ≤ 2 ГБ и 30 минут на таком регионе. Crop verify на реальных кропах не проверялся (кропов на коробке нет). GPU и платформенный образ. Лимиты 20000 и 3000 гарантированы кодом: `finalize`/`limit_vertices`, `cap_features` и новый `enforce_limits`.

## 4. Честный замер на коробке (без crop), unseen keep, 10 фолдов по семьям
Вход `work2/fam_kur` (стек bot-10 + курганы 0,5 м) → `scripts/pkg_fam.py` (stack6 separate [+twins]) → `family_curves` → `honest_bootstrap --unseen keep`.
| вариант | honest10 | Δ к базе 0,333237 | P | CI 95 % | LOFO Δ | 5 раскладок |
|---|---|---|---|---|---|---|
| kur (REPORT2) | 0,334608 | +0,001371 | — | — | — | — |
| **pkg = stack6 + kur** | 0,349665 | **+0,016428** | 0,994 | [+0,00026; +0,02114] | +0,015121 | 5/5 (+0,0128…+0,0163) |
| pkg − kur (= stack6) | | +0,015057 | 0,948 | | +0,015057 | 5/5 |
| **pkgT = pkg + twins_only** | 0,350017 | +0,016780 | 1,000 | [+0,00114; +0,02159] | +0,015313 | 5/5 |
| **pkgT − pkg (twins)** | | **+0,000352** | 0,973 | [−0,00000; +0,00135] | +0,000192 | 4/5 (одна −0,00019) |
Twins по классу: только kurgany_povrezhdennye, tp +5, fp +9. Это заметно меньше, чем +0,00285 у полного варианта bot-6. Есть оговорка: здесь копии мерены до порогов со своим порогом группы, а в сдаче они делаются после порогов. stack6 + kur складываются точно (+0,015057 + 0,001371 = +0,016428): классы не пересекаются. Файлы: `work4/hb_pkg.json`, `hb_pkgT.json`, логи `work4/fam_pkg*.log`, `curves_pkg*.log`. Время: pkg_fam около 6 мин, curves 205 / 177 с, бутстрэп около 250 с. Посемейные стеки удалены (воспроизводимы).

## 5. Ожидаемый прирост
Аддитивно по классам: stack6 трогает object_poly и finds, kur/twins — курганы, R — только архитектуру.
- Без crop: +0,016428 (замер) + R +0,009536 (bot-10 strict, только arkhitektury) ≈ **+0,026** к honest-10 базы. С поправкой bot-10 на отбор R +0,0035 получается ≈ +0,020.
- Crop verify: +0,00414 отдельно, но с kur он не аддитивен (bot-10). Итог с crop — **только после замера на ПК**.
- Twins: +0,00035, на грани шума. Решать после перемера.

## 6. Команды для ПК (PowerShell, только C:; F: не читается)
Скрипт `bot-7/work4/pc/pc_measure_pkg.ps1`. Набор для переноса — `work4/pc/` (`candidate_measure.mbox`, `family_clusters.json`, `SHA256SUMS`). В начале скрипта проверка: любой путь не с `C:\` → остановка, нет входного файла → остановка. Двух веток нет: twins включается флагом `-Twins` (`pkg_fam.py --twins` — то же, что `[post] kurgan_twins_only = true`).

Без twins:
```
powershell -ExecutionPolicy Bypass -File C:\expds3c\bot7\pc_measure_pkg.ps1
```
С twins (`kurgan_twins_only = true`):
```
powershell -ExecutionPolicy Bypass -File C:\expds3c\bot7\pc_measure_pkg.ps1 -Twins
```
Шаги скрипта:
1. worktree `C:\expds3c\wt_bot7` = `24bd916` + `git am candidate_measure.mbox`.
2. `split_stack_by_family` для базы и для стека crop.
3. `micro_fam` (курганы 0,5 м).
4. `pkg_fam [--twins]`.
5. `family_curves`.
6. `honest_bootstrap --unseen keep` против `base` и `crop`. Выход `C:\expds3c\pkg\hb_pkg[T].json`, время в `time_pkg[T].txt`.

Проверка уровня: `base` должен дать honest10 0,333237. R прибавляется из bot-10 (+0,009536), потому что классы не пересекаются.

**Что лежит на F: и как перенести на C:** (после окончания переноса, пользователем):
- `F:\expds3\work\stack_verify_sp.geojson` — стек crop verify. **Есть только на F:.** Имя взято из docstring `scripts/crop_verify.py` и является допущением; другое имя передаётся через `-Crop <имя>`. Перенос: `robocopy F:\expds3\work C:\expds3c\in stack_verify_sp.geojson /J`.
- `stack_base.geojson` — стек базы. Можно с F: (`robocopy F:\expds3\work C:\expds3c\in stack_base.geojson /J`) или **без F:** с коробки: `/workspace/rerank/stack_arch.geojson.gz` (193 МБ, распаковать в `C:\expds3c\in\stack_base.geojson`). Это ровно тот стек, на котором bot-10 получил 0,333237.
- `gt_train_v2.geojson` — с коробки `/workspace/rerank/gt_train_v2.geojson` (sha256 `d4bc1a3b…ff86`), или `robocopy F:\expds3\work C:\expds3c\in gt_train_v2.geojson /J`.
- `--train F:\2\train-expds-3`, OOF `F:\expds3\work\*_oof` и `mosaic_li` для этого замера **не нужны**: scoreboard и crop_verify заново не запускаются.

Допущения: репо `C:\Sloi\expds-koz3_v0.2.0\expds-koz3` с `.venv` (shapely, scipy) и коммит `24bd916` в нём; стек crop снят с теми же флагами базы, что и `stack_base`. Если `base` ≠ 0,333237 — числа не сравнивать.

## 7. Риски
1. Порядок twins: в сдаче копии делаются после порогов, в замере — до порогов, со своим порогом группы.
2. RSS 2291 МБ на объединённом прокси из 6 семей. Если реальный регион окажется сопоставим, лимит в 2 ГБ под угрозой. На ПК стоит прогнать полный регион через `check_submission --input`.
3. Реранк учили на уверенностях до crop; пакетом они не мерены (§6 закроет только не-архитектурную часть).
4. RSS и время полного main.py на крупном регионе не мерены.
5. Архив собирать сборщиком репо или `build_submission_git` (bot-8, теперь с `[rerank_arch]`).

## 8. Файлы
- `bot-7/candidate.diff` (sha256 `81673ccb…3958`), `bot-7/candidate.mbox` (`4c312ade…ce4a`): `24bd916..bot7-candidate`, 20 файлов, +8093/−18.
- `bot-7/measure_tools.diff` (`d5209fa1…868a`): `bot7-candidate..bot7-candidate-measure` (`1b0c294`: инструменты bot-10 `caae8cc`, `31331f6`, `micro_fam` `b42f289`, `pkg_fam` `1b0c294`). Только для замера, не для сдачи.
- `work4/pc/` — скрипт ПК и набор для переноса; `work3/` — zip, логи check, pytest, toml_diff.
