# bot-7 · REPORT3 · кандидат на посылку (28.10.2026, МСК)

## Вывод (3 строки)
1. Ветка `bot7-candidate` (HEAD `fe0e570`, локально, без push) = база `24bd916` + курганы 0,5 м (без микрокругов архитектуры) + исправленный корень весов crop verify (bot-8) + реранк архитектуры, **включён** одним флагом `[rerank_arch] enabled = true`.
2. В конфиге изменилось ровно 5 ключей (`work3/toml_diff.txt`). Тесты: 315 passed, 4 skipped. Архив собран сборщиком репозитория, `check_submission` (bot-8) с дымовым прогоном: PASS 28/28 (`work3/check.log`).
3. Пакетом с crop verify реранк не мерили: OOF лежат на F:, и он занят. Ожидаемый прирост складывается из отдельных замеров и не проверен.

## 1. Ветка и коммиты (`/workspace/koz3_shared/bot-7/repo`, база `24bd916`)
| коммит | что меняет |
|---|---|
| `42ada8d` | bot7-bundle: код `post.micro_circles` (радиус по классу), хук в `watchdog.finish`, `scripts/oof_transform.py micro`. Блок конфига в этом коммите закомментирован, сам по себе он ничего не меняет. В сообщении упомянута архитектура 0,25 м, но в кандидат она не входит, см. `dd7982e` |
| `07bed29` | cherry-pick bot-8 `8343483`: `[verify] weights` ищутся от корня архива, а не от cwd. Добавлен `tests/test_bot8_crop_root.py`. При конфликте modify/delete отброшен чужой `test_bot8_faults.py`. Коммит bot-8 `9316b5f` (надёжность) сюда **не** взят, его лучше смотреть отдельно |
| `dd7982e` | `[post.micro_circles] radius_m = { kurgany_povrezhdennye = 0.5, kurgany_tselye = 0.5 }`, ключа `arkhitektury` нет. Пороги не менялись: `sweep_groups` в выборке по OOF после micro дал те же пороги, что у базы (`work3/insample_tau_kur.json`; база 0,365386 → 0,366736 в выборке); `or.arkhitektury` = 0,065565 |
| `733d3ed` | код флага: `expds/rerank_arch.py`, хук в `watchdog.finish` (после `only_known_classes`, до `apply_thresholds`; затем merge и micro), `scripts/rerank_arch_train.py`. В `scripts/build_submission.py` файлы модели кладутся в архив, если флаг включён. Тесты `tests/test_bot7_rerank_arch.py`. Флаг в этом коммите выключен |
| `fe0e570` | `[rerank_arch] enabled = true`, `model = "models/rerank_arch_lgbm.txt"`, `meta = "models/rerank_arch_lgbm.json"`. Модель обучена на всех пулах архитектуры (`work3/train.log`). sha256: txt `0bc2e924b7a7beb79706eae275d516a2668406294e99b6caf44e454030b6d97b`, json `d76c7edc54fc1c377a652122dcc8c7022e62244d48a6c0bbe85ecc19d2215ab0` |

Дерево HEAD: `e75e3116851b028c8110fe32b3f0f7e3ed7a2806`. После `git am` на ПК хеши коммитов будут другими, а хеш дерева должен совпасть.
Прежний вариант (микрокруги архитектуры, порог `or` 0,060125, реранк выключен) оставлен в ветке `bot7-candidate-v1` (`913e2be`). Его zip/diff/mbox удалены.

**Флаг:** `[rerank_arch] enabled`, по умолчанию в кандидате `true`. **Выключить:** одна строка `enabled = false`. Если флаг включён, а файлов модели нет или они не читаются, код пишет warning или exception в лог и возвращает объекты без реранка (`expds/rerank_arch.py:191-204`).

Плоский diff конфига `24bd916 → HEAD` (`work3/toml_diff.txt`), больше изменений нет:
```
post.micro_circles.radius_m.kurgany_povrezhdennye : <нет> -> 0.5
post.micro_circles.radius_m.kurgany_tselye : <нет> -> 0.5
rerank_arch.enabled : <нет> -> True
rerank_arch.meta : <нет> -> models/rerank_arch_lgbm.json
rerank_arch.model : <нет> -> models/rerank_arch_lgbm.txt
```
Crop verify уже включён в базе `24bd916`, его ключи не менялись. Fix bot-8 правит только поиск файла весов.

## 2. Смена состава по просьбе бота 1 и доказательства
- **Почему без микрокругов архитектуры.** Источник: `bot-5/REPORT.md`, раздел «ОБНОВЛЕНИЕ 28.10, 20:56 МСК», данные `bot-5/analysis/joint/summary.json`, скрипт `bot-5/work/joint_rerank_micro.py`. На канонической раскладке реранк R даёт +0,00974, R+micro даёт +0,01043, разница RM−R +0,00069 (P 0,859). На seed3 RM−R = −0,00345 (P 0,173), в среднем по 5 раскладкам RM−R = −0,00035. Вывод bot-5: брать R, micro не добавлять. **Подтверждено файлами.**
- **Почему реранк включён.** EXPERIMENTS.md, строка rerank (исполнитель rerank): +0,00974, P 0,966, П1–П4 ✓, с пометкой «перемерить пакетом с crop verify». Перемер bot-10 есть только в сырых логах `bot-10/work4/rerank/{k10,lofo,seed0..4}.log` (`nested_lofo5.py`, вариант `full_orig`): k10 +0,009536 (P 0,957, [+0,00063; +0,01535]), LOFO +0,007869 (P 0,934, [−0,00053; +0,01394]), раскладки seed0–4: +0,00241 / +0,00844 / +0,01051 / +0,00627 / +0,00739, то есть 5 из 5 > 0 (у seed0 P 0,80). Варианты `noRG2` слабее, `noRGall` уходит в минус. Прогоны `*_site` на момент чтения не закончились (`lofo_site.log` содержит только заголовок). **Цифра «+0,008–0,010» подтверждается логами bot-10, но строки EXPERIMENTS и отчёта bot-10 по ней нет.**
- **Курганы 0,5 м без пакета:** +0,00137, P 0,953 (REPORT2, `work2/hb_kur.json`). По П1 это ниже порога, поэтому они взяты по решению бота 1. Учти это при приёмке.

## 3. Ожидаемый прирост
Относительно `24bd916` (crop verify уже в базе: +0,00414, P 0,984, EXPERIMENTS строка 33):
- реранк: около +0,008–0,0095 по честным замерам выше;
- курганы: +0,0014.

Сумма около **+0,009–0,011** — **оценка, не замер**. Реранк учили на уверенностях до crop verify, а пакетом с verify его не мерили.

## 4. Проверки на коробке (2 потока, nice 10)
| проверка | результат | файл |
|---|---|---|
| pytest всего репо (`-k "not coco_roundtrip"`, тесту нужен cv2, его в venv нет; на базе он падает так же) | 315 passed, 4 skipped, 3 deselected | `work3/pytest_repo.log` |
| diff конфига | только 5 ключей выше | `work3/toml_diff.txt` |
| сборка `scripts/build_submission.py` (сборщик репо) | `work3/dist/submission_bot7cand_fe0e570.zip`, 523 МБ, sha256 `f1c7670ddafbc6ef30318070544342dfcfb59551676856757ff5c0f46a3f9414`; оба файла `models/rerank_arch_lgbm.*` в архиве | `work3/build.log` |
| `bot-8/check_submission.py` с дымовым прогоном (`--lock repo/requirements.lock`) | **PASS, FAIL 0, WARN 0, 28 проверок**. main.py 8,8 с, пик RSS 973 МиБ; в логе «реранк архитектуры: 2 кандидатов, в пуле 2, оставлено 2»; 80 полигонов | `work3/check.log` |
| время `rerank_arch.apply` на реальной семье стека: Курганы + Вышегжа, 206 050 объектов, 37 регионов, 12 379 архитектуры | 6,6 с, пик RSS 697 МБ (2 потока), оставлено 6 строк архитектуры. На Вышегже без архитектуры 0,3 с | `work3/time_rerank.{py,log}` |
| сборщик bot-8 (`bot-8/build_submission.py`) | **не годится**: `model_refs` берёт только `[models]`, `candidates.lgbm_model`, `verify.weights` (строки 51-68), файлов реранка в архиве не будет, и check FAIL «файлы из default.toml отсутствуют» (видно на v1: `work3/build_on.log`). Собирать **сборщиком репо** | — |

**Не проверено:** честный замер всего пакета с crop verify (нужны OOF с `F:\expds3\work`); GPU и полный регион на платформенном образе; `platform_sim` на `mock_val` (лежит на F:); LB.

## 5. Команды для ПК (Windows, PowerShell, **только C:**)
Допущения: репо `C:\Sloi\expds-koz3_v0.2.0\expds-koz3` и `.venv` (`docs/AI_WORK_PLAN_2026-10-25.md`); на ПК HEAD = `24bd916`; пользователь сам переносит `bot-7/candidate.mbox` в `C:\Sloi\bot7\candidate.mbox` и (по желанию) `bot-7/work3/synth_input\` в `C:\Sloi\bot7\synth_input\`. Пути `C:\Sloi\bot7\...` придуманы как места для копий и **являются допущением**.
```powershell
cd C:\Sloi\expds-koz3_v0.2.0\expds-koz3
git status --short                      # должно быть пусто
git rev-parse HEAD                      # ожидается 24bd916e6adff128c9ab8b22fa53f792881080cb
git checkout -b bot7-candidate 24bd916
git -c core.autocrlf=false am --keep-cr C:\Sloi\bot7\candidate.mbox
git rev-parse "HEAD^{tree}"             # ожидается e75e3116851b028c8110fe32b3f0f7e3ed7a2806
certutil -hashfile models\rerank_arch_lgbm.txt SHA256   # 0bc2e924...d97b
$env:OMP_NUM_THREADS = "2"
.venv\Scripts\python -X utf8 -m pytest -q -p no:cacheprovider tests\test_bot7_rerank_arch.py tests\test_bot8_crop_root.py
.venv\Scripts\python -X utf8 scripts\build_submission.py --out dist\submission_bot7cand.zip
# по желанию, локальный прогон архива на синтетике с C: (образ из metadata.json; что он есть локально — допущение):
powershell -File scripts\platform_sim.ps1 -Image pantum/n7q:20261024 -Data C:\Sloi\bot7\synth_input -Out C:\Sloi\bot7\platform_sim -Zip dist\submission_bot7cand.zip
```
Зависимости от F: и что с ними делать:
- `platform_sim` на `F:\expds3\mock_val` (пример в `scripts/platform_sim.ps1:13` пишет и Out на F:): **заблокировано**. Замена только на C: — синтетика `synth_input` (3,7 МБ) и Out на C:. Это проверка «запускается и пишет ответ», а не проверка качества.
- Честный перемер пакетом с crop verify (`scoreboard.py --work F:\expds3\work --train F:\2\train-expds-3 ...`, `docs/AGENT_PLAN.md:35-44`): **заблокировано**, эталон, OOF и выборка лежат только на F:. Обходного пути через C: нет.
- Docker Desktop. Где он хранит образы, по документам неизвестно. Если это F:, `platform_sim` тоже заблокирован. **Не проверено.**
- `check_submission.py` (bot-8) в репо не входит, а его дымовой тест рассчитан на Linux (`unshare`/`taskset`). На ПК он нужен только при переносе файла; запускать с `--no-smoke` (допущение). Полная проверка уже прошла на коробке (п. 4).
- Посылка на LB решается человеком; я её не делал.

## 6. Риски
1. Реранк учили на уверенностях до crop verify, вместе их не мерили. Возможна двойная фильтрация архитектуры.
2. Финальная модель обучена на всех данных, и её калибровка отличается от фолдовых OOF-моделей (tau: `or` 0,001373, `sp_arch` 0,019047). Пул и `gq` считаются по распределению теста.
3. Подтверждение bot-10 есть только в сырых логах, без строки EXPERIMENTS; на seed0 P 0,80; LOFO по площадкам не досчитан.
4. Курганы 0,5 м поодиночке ниже П1 (P 0,953).
5. Реранк добавляет секунды к `finish`. `finish` вызывают сторож (с запасом до бюджета) и SIGTERM. На 206 тыс. объектов это 6,6 с на CPU; запас сторожа, по моим допущениям, его покрывает, но на платформе не проверено.
6. Архив нужно собирать **сборщиком репо**, а не bot-8. Иначе реранк молча выключится (warning, база), а check выдаст FAIL.
7. В v2 кандидате нет `9316b5f` bot-8 (надёжность). Его рекомендую рассмотреть отдельно.

## 7. Файлы
- `bot-7/candidate.diff` (sha256 `2a70d46a69d97f44090f616f86ff133e10e0658772d3b69386ec61747440df8b`), `bot-7/candidate.mbox` (`0d5fb229a88093e208ca530c7046f61f9708f708ba7a840ef32deef8bf74553f`): `24bd916..bot7-candidate`, 12 файлов, +6518/−2.
- `bot-7/work3/`: `dist/submission_bot7cand_fe0e570.zip`, `build.log`, `check.log`, `pytest_repo.log`, `toml_diff.txt`, `time_rerank.{py,log}`, `insample_tau_kur.{py,json}`, `train.log`, `synth_input/`.
