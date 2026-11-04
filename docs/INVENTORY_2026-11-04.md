# Перепись артефактов KOZ3 после блокировки Kaggle (04.11.2026)

04.11 Kaggle заблокировал 4 из 5 аккаунтов команды: `elenakonkova`, `usertest1234`, `aaaa0000004`, `aaaadg4gg4g4g`. При входе Kaggle показывает «Your account has been suspended or banned». Через API профиль отдаёт 404, ответ «users.get denied» такой же, как у несуществующего пользователя, токены возвращают 401. Ниже — где что лежит теперь, что пропало и чем это закрыто. Проверка повторяема: `scripts/inventory_lost_accounts.py`.

## 1. Где что лежит

| Место | Что | Объём | Проверка |
|---|---|---|---|
| GitHub `KonkovDV/expds-koz3` | код, конфиги, веса сдачи `models/*.pt`, доки, PREREG и результаты гейтов | `main`: 3346 файлов; пакет `.git` 816 МБ; 39 веток | 37 веток целиком в `main`; у `bots/orch-protocol` только коммиты слияния, своих правок 0 строк; `git fsck` чисто |
| Google Drive «2» (`adincris2018@gmail.com`, 5,5 ТБ) → `KOZ3_backup_2026-11-04` | **основная копия всего**, см. §2 | см. §2 | md5 каждого файла сверен с Drive при выгрузке |
| Google Drive `KOZ3-final` (`99999kupor@gmail.com`) | посылки v4 / v3_on / v2clean, бэкапы 02.11, отчёты; `backup_2026-11-04/` (01, 02 и доки) | 5,9 ГБ | md5 при копировании |
| Google Drive `KOZ3-pixgbm` (`99999kupor@gmail.com`) | сборки sat2/fr1h/pixgbm от 01.11 | 4,4 ГБ | md5 при копировании |
| Kaggle `dmitrijkv` (единственный живой аккаунт) | 58 ядер и их выводы; 22 приватных датасета, в том числе `koz3-gate-box-20261104` и `koz3-handoff-20261104` | ~25 ГБ | перенос в §2, папка 07 |
| Бокс `/workspace/koz3` (облачный контейнер, временный) | восстановленный бокс гейтов | 5,7 ГБ | целиком в §2, папки 01, 03, 05, 06 |

## 2. Google Drive «2» → `KOZ3_backup_2026-11-04`

| Папка | Что внутри | Объём |
|---|---|---|
| `00_README.md` | что где лежит и как восстановить бокс без Kaggle | — |
| `01_gate_base` | `stack_arch.geojson` (стек OOF базы), `gt_train_v2.geojson`, `table.pkl`, `base_counts.pkl` — tar.xz из 2 частей + `SHA256SUMS`, суммы файлов внутри архива | 132 МБ |
| `02_dfine_spor_folds` | веса фолдов f0–f4 прод-модели D-FINE спутника и отчёты (tar, 7 частей) | 625 МБ |
| `03_oof` | OOF D-FINE (`w4geo/spor_oof`), D-FINE-2 (`w2/oof/dfine2`), `sat_cl2` — tar.xz | 328 МБ |
| `04_lidar` | фолды лидара: `li_r18_hard2` f0–f4 + final (прод-модель курганов), `li_r34_fpn` mask/edge, `centers_fix`; кэш тайлов `li-tin-59-t128`; кропы курганов | 2,6 ГБ |
| `05_tta_v2` | выход ядра TTA D-FINE v2: V0/T8N/T8W по фолдам, отчёт | 2,4 ГБ |
| `06_box_rest` | остаток бокса без того, что лежит в 01/03/05 и без venv — tar.xz | см. `SIZE.txt` |
| `07_kaggle_dmitrijkv` | `kernels/<slug>/`: вывод, исходник, лог каждого ядра; `datasets/<slug>/`: файлы всех датасетов | ~25 ГБ |
| `08_KOZ3-final_mirror` | серверная копия `KOZ3-final`. 8 частей v4, лежавших по два раза с одинаковым md5, — по одной | 5,9 ГБ |
| `09_KOZ3-pixgbm_mirror` | серверная копия `KOZ3-pixgbm`. Одноимённые части двух разных сборок sat2 разведены суффиксом с датой создания | 4,4 ГБ |
| `10_github_repo` | `git bundle` всех 39 веток с историей; `REFS.txt` | 842 МБ |

## 3. Что было на заблокированных аккаунтах

Всего в git 127 ссылок на ресурсы этих аккаунтов. Слоты аренды вроде `cpu` или `gpu-1` в это число не входят. Ниже ресурсы сгруппированы по смыслу. Полный список ссылок — в приложении (вывод скрипта).

| Что | Ресурсы | Статус | Где копия / почему не нужна |
|---|---|---|---|
| **Стек базы гейта** | `elenakonkova/koz3-w2-gate` | **закрыто** | `dmitrijkv/koz3-gate-box-20261104` и Drive «2»/01: копия из бокса с sha256 |
| **Архив бокса и handoff 02.11** | `<все>/koz3-handoff-20261102(b)` | закрыто | `dmitrijkv/koz3-handoff-20261102`, Drive `backup_2026-11-02`, «2»/08 |
| **Архив посылки v4** | `aaaa0000004/koz3-w4-v4c1f` | закрыто | Drive `v4c1f_f0def982` (8 частей), «2»/08; sha256 сверен 03.11; v4 пересобирается из `main` побайтно |
| Веса лидара ConvNeXt (W1/W2) | `elenakonkova/koz3-cnx-li-*`, `koz3-w2-cnx-c3sc-*`, `koz3-w2-lihm-c3sc-f01234`, `koz3-w2-cnx-r50c5-*` | закрыто частично | веса семян — в `koz3bk_W1_lidar_cnx_seeds` (handoff, Drive `backup_2026-11-02/weights`); линии r50c5 и ConvNeXt закрыты гейтами |
| D-FINE ортофото и аэрофото | `elenakonkova/koz3-w1-dfine-or-final`, `-ae-final` | or — закрыто, ae — утеряно | or — в `koz3bk_W2_dfine_or_final`; ae — линия `ae-dfine-gate` закрыта |
| Сеть курганов на CPU (W3) | `<все>/koz3-w3-kurgan-cpu-seeds(-b)`, `-data` | закрыто | `koz3bk_W3_kurgan_cpu` (handoff), `dmitrijkv/koz3-w3-kurgan-cpu-data` |
| Признаки геофизики (W3) | `koz3-w3-geo5-*` на четырёх аккаунтах | закрыто частично | база — `dmitrijkv/koz3-w3-geo5-feat-base`; варианты n40r8/l-s* утеряны, их линии закрыты |
| Faster R-CNN спутника | `elenakonkova/koz3-w1-satdet-frcnn` | утеряно | линия закрыта; гейт W1 требовал `prep.pkl`, которого нет ни в одном бэкапе |
| Проверки паритета сборок (W4) | `koz3-w4-v2clean-check`, `-v3onchk`, `-v4chk`, `-v3amin-chk`, `-fr1h-check`, `-v3onfix` | утеряно, не нужно | итоги записаны в `orch/FINAL_CANDIDATES.md` и `orch/bot-W4/STATUS_W4.md`; для новой сборки паритет считается заново |
| Колёса pip для сборок | `aaaa0000004/koz3-w4-wheels(-v3)` | утеряно, восстановимо | собираются с PyPI; вендор лежит внутри архивов посылок |
| Данные участков (копии S3) | `koz3-w4-dfine-ref103`, `koz3-gt-train-v2`, `koz3-w1-sat2-src`, `koz3-wsr-data`, `koz3-li-aug-data` | утеряно, восстановимо | первоисточник — бакет организаторов `train-expds-3`; эталон есть в «2»/01 |
| Продукты ЦМР лидара (W2) | `elenakonkova/koz3-w2-demfill`, `-relief`, `-maps`, `-weights`, `-oof`, `-hmcp-*` | частично в боксе | части `w2/` бокса — в «2»/06; остальное пересчитывается из LAS |
| DINO и облачные признаки (W5) | `koz3-w5-dino-*`, `koz3-dino-gor-*`, `koz3-w5-cloudfeat(-in)`, `koz3-w5-cands` | утеряно | `dino-gor` закрыт; `w5_cloudfeat` в коде за флагом OFF, не в v4 |
| Закрытые опыты | `koz3-lidar-aug-f0`, `koz3-wsr-f0`, `koz3-peak-rerank-*`, `koz3-w4-large-gate*`, `koz3-w4-li-post-*`, `koz3-sp-large-f0` | утеряно, не нужно | итоги в `docs/STATUS.md` / `docs/EXPERIMENTS.md`; `sp-large-f0(-data)` есть на `dmitrijkv` |
| Учётки S3 | `elenakonkova/koz3-s3-cred` | утеряно | ключи у владельца; на Kaggle больше не класть |
| Инфраструктурные пробы | `koz3-acc2-tpu-smoke`, `koz3-w2-gaudit-las`, `koz3-w2-ground-clip-audit` | утеряно, не нужно | — |

**Итог.** Всё, без чего нельзя считать гейты или собрать посылку, сохранено: стек базы, OOF D-FINE и D-FINE-2, веса фолдов D-FINE и лидара, архив бокса, посылки v4/v3_on/v2clean, весь код. Утеряны выводы закрытых опытов и проверок паритета (их итоги записаны в доках), а также копии данных организаторов, которые восстанавливаются из S3.

## 4. Как не потерять дальше

- Kaggle: работать только на `dmitrijkv`, новых аккаунтов не создавать. Правило Kaggle — один аккаунт на человека.
- Любой новый артефакт, будь то OOF, веса или выход ядра, сразу класть в Drive «2» (`KOZ3_backup_2026-11-04` или папку с датой) с SHA256SUMS. Kaggle — не место хранения.
- `scripts/box_restore.sh` восстанавливает бокс с `dmitrijkv`. Порядок восстановления без Kaggle — в `00_README.md` на Drive.

## Приложение. Все ссылки на ресурсы заблокированных аккаунтов (`scripts/inventory_lost_accounts.py`)

Колонки «на Drive» и «в боксе» означают, что имя ресурса встречается среди имён файлов. Это подсказка, а не доказательство копии: совпадения по содержимому сведены в §3.
| аккаунт | ресурс | упоминаний | копия на dmitrijkv | на Drive | в боксе | первое упоминание |
|---|---|---|---|---|---|---|
| elenakonkova | `koz3-lidar-aug-f0` | 91 | — | — | да | `HANDOFF/INFRA.md`: elenakonkova/koz3-lidar-aug-f0 koz3-lidar-aug-f0 Elena Konkova 2026-1 |
| elenakonkova | `koz3-w2-gate` | 45 | — | — | — | `HANDOFF/INFRA.md`: - dmitrijkv can't read elenakonkova/koz3-w2-gate → gates needing it run on elenakonkova. |
| elenakonkova | `koz3-w2-cnx-r50c5-f01` | 33 | да | — | да | `HANDOFF/INFRA.md`: elenakonkova/koz3-w2-cnx-r50c5-f01 koz3-w2-cnx-r50c5-f01 Elena Konkova 2026-1 |
| elenakonkova | `koz3-acc2-tpu-smoke` | 30 | — | — | — | `HANDOFF/INFRA.md`: elenakonkova/koz3-acc2-tpu-smoke koz3-acc2-tpu-smoke Elena Konkova 2026-1 |
| aaaa0000004 | `koz3-w4-v2cleanchk2` | 29 | — | — | да | `HANDOFF/INFRA.md`: aaaa0000004/koz3-w4-v2cleanchk2 koz3-w4-v2cleanchk2 AA AA0000004 2026-1 |
| elenakonkova | `koz3-s3-cred` | 28 | — | — | — | `HANDOFF/INFRA.md`: - Private cred datasets: dmitrijkv/koz3-s3-cred, elenakonkova/koz3-s3-cred (GPU accounts only). |
| elenakonkova | `koz3-w2-lihm-c3sc-f01234` | 26 | — | — | да | `HANDOFF/INFRA.md`: elenakonkova/koz3-w2-lihm-c3sc-f01234 koz3-w2-lihm-c3sc-f01234 Elena Konkova 2026-1 |
| elenakonkova | `koz3-w1-satdet-frcnn` | 23 | — | — | — | `CHANGELOG.md`: - На Kaggle не собраны (статус complete 02.11): `dmitrijkv/koz3-w2-dfine2-{a,b}` (D-FINE SpOr, 5 фолдов) и `elenakonkova/koz3-w1-satdet-frcn |
| elenakonkova | `koz3-w2-demfill` | 22 | — | — | — | `HANDOFF/INFRA.md`: elenakonkova/koz3-w2-demfill koz3-w2-demfill 123247104 2026-10-30 09:1 |
| elenakonkova | `koz3-w3-gordense-timing-k` | 22 | — | — | — | `HANDOFF/INFRA.md`: elenakonkova/koz3-w3-gordense-timing-k koz3-w3-gordense-timing-k Elena Konkova 2026-1 |
| elenakonkova | `koz3-w5-dino-vitb-tpu` | 22 | да | — | — | `HANDOFF/INFRA.md`: elenakonkova/koz3-w5-dino-vitb-tpu koz3-w5-dino-vitb-tpu Elena Konkova 2026-1 |
| aaaa0000004 | `koz3-w2-cnx-r50c5-f01` | 20 | да | — | да | `HANDOFF/INFRA.md`: aaaa0000004/koz3-w2-cnx-r50c5-f01 koz3-w2-cnx-r50c5-f01 AA AA0000004 2026-1 |
| elenakonkova | `koz3-w2-oof` | 19 | — | — | — | `EXPERIMENTS.csv`: 2026-11-01,bot4,bots/bot4-li-post,li_post OFF (tau=0.15),honest-LOFO-s0-cnx Kaggle COMPLETE,elenakonkova/koz3-w2-oof via koz3-w4-li-post-gat |
| elenakonkova | `koz3-w3-kurgan-cpu-seeds-b` | 19 | — | — | — | `HANDOFF/INFRA.md`: elenakonkova/koz3-w3-kurgan-cpu-seeds-b koz3-w3-kurgan-cpu-seeds-b Elena Konkova 2026-1 |
| aaaa0000004 | `koz3-w2-gaudit-las` | 18 | — | — | — | `HANDOFF/INFRA.md`: aaaa0000004/koz3-w2-gaudit-las koz3-w2-gaudit-las 3466757301 202 |
| elenakonkova | `koz3-cnx-li-c3sc-s1f23` | 18 | — | — | да | `HANDOFF/INFRA.md`: elenakonkova/koz3-cnx-li-c3sc-s1f23 koz3-cnx-li-c3sc-s1f23 Elena Konkova 2026-1 |
| elenakonkova | `koz3-cnx-li-prep` | 18 | да | — | — | `HANDOFF/INFRA.md`: elenakonkova/koz3-cnx-li-prep koz3-cnx-li-prep 799473915 2026-10-29 17:0 |
| usertest1234 | `koz3-w4-v2clean-check` | 18 | — | — | — | `HANDOFF/INFRA.md`: usertest1234/koz3-w4-v2clean-check koz3-w4-v2clean-check 526506473 20 |
| elenakonkova | `koz3-cnx-li-c3sc-s1f4` | 16 | — | — | да | `orch/COORD_LOG.md`: - 29.10 22:23 MSK / gpu_watcher / acc1/gpu-1: elenakonkova/koz3-cnx-li-c3sc-s1f4 → KernelWorkerStatus.COMPLETE; аренда W1; push elenakonkova |
| elenakonkova | `koz3-w2-cnx-c3sc-s2f4` | 16 | — | — | да | `HANDOFF/INFRA.md`: elenakonkova/koz3-w2-cnx-c3sc-s2f4 koz3-w2-cnx-c3sc-s2f4 Elena Konkova 2026-1 |
| elenakonkova | `koz3-w1-dfine-or-final` | 14 | — | — | да | `HANDOFF/INFRA.md`: elenakonkova/koz3-w1-dfine-or-final koz3-w1-dfine-or-final Elena Konkova 2026-1 |
| elenakonkova | `koz3-w5-dino-emb-gpu` | 14 | — | — | да | `HANDOFF/INFRA.md`: elenakonkova/koz3-w5-dino-emb-gpu koz3-w5-dino-emb-gpu Elena Konkova 2026-1 |
| aaaa0000004 | `koz3-w2-ground-clip-audit` | 13 | — | — | — | `HANDOFF/INFRA.md`: aaaa0000004/koz3-w2-ground-clip-audit koz3-w2-ground-clip-audit AA AA0000004 2026-1 |
| elenakonkova | `koz3-w1-dfine-ae-final` | 13 | — | — | да | `HANDOFF/INFRA.md`: elenakonkova/koz3-w1-dfine-ae-final koz3-w1-dfine-ae-final Elena Konkova 2026-1 |
| elenakonkova | `koz3-w4-large-gate-code` | 13 | — | — | — | `HANDOFF/INFRA.md`: elenakonkova/koz3-w4-large-gate-code koz3-w4-large-gate-code 2678241 2026-11-01 17:2 |
| aaaa0000004 | `koz3-w3-geo5-base-s12` | 12 | — | — | — | `HANDOFF/INFRA.md`: aaaa0000004/koz3-w3-geo5-base-s12 koz3-w3-geo5-base-s12 AA AA0000004 2026-1 |
| aaaa0000004 | `koz3-w3-kurgan-cpu-seeds` | 12 | да | — | да | `HANDOFF/INFRA.md`: aaaa0000004/koz3-w3-kurgan-cpu-seeds koz3-w3-kurgan-cpu-seeds AA AA0000004 2026-1 |
| elenakonkova | `koz3-dino-gor-raw` | 12 | — | — | — | `HANDOFF/INFRA.md`: elenakonkova/koz3-dino-gor-raw koz3-dino-gor-raw 3709636895 2026-10-31 19:0 |
| elenakonkova | `koz3-w2-weights` | 12 | — | — | — | `HANDOFF/INFRA.md`: elenakonkova/koz3-w2-weights koz3-w2-weights 1358396794 2026-10-30 09:0 |
| elenakonkova | `koz3-wsr-f0` | 12 | — | — | — | `HANDOFF/INFRA.md`: elenakonkova/koz3-wsr-f0 koz3-wsr-f0 Elena Konkova 2026-1 |
| usertest1234 | `koz3-w4-v2cleanchk-k` | 12 | — | — | да | `HANDOFF/INFRA.md`: usertest1234/koz3-w4-v2cleanchk-k koz3-w4-v2cleanchk-k User Test1234 |
| aaaa0000004 | `koz3-w4-v3onchk` | 11 | — | — | да | `HANDOFF/INFRA.md`: aaaa0000004/koz3-w4-v3onchk koz3-w4-v3onchk AA AA0000004 2026-1 |
| elenakonkova | `koz3-w3-gordense-timing` | 11 | да | — | да | `HANDOFF/INFRA.md`: elenakonkova/koz3-w3-gordense-timing koz3-w3-gordense-timing 793441661 2026-10-31 08:0 |
| elenakonkova | `koz3-w5-cloudfeat` | 11 | — | — | да | `HANDOFF/INFRA.md`: elenakonkova/koz3-w5-cloudfeat koz3-w5-cloudfeat Elena Konkova 2026-1 |
| aaaa0000004 | `koz3-w4-v2clean-check` | 10 | — | — | — | `HANDOFF/INFRA.md`: aaaa0000004/koz3-w4-v2clean-check koz3-w4-v2clean-check 526506643 202 |
| aaaa0000004 | `koz3-w4-v4c1f` | 9 | — | — | — | `orch/COORD_LOG.md`: - 02.11 14:59 MSK / lease / acquire dataset:aaaa0000004/koz3-w4-v4c1f → bot-W4-exec до 03.11 14:59 MSK v4 parity data (private) |
| aaaadg4gg4g4g | `koz3-w5-cloudfeat` | 9 | — | — | да | `orch/COORD_LOG.md`: - 30.10 09:04 MSK / lease / acquire kernel:aaaadg4gg4g4g/koz3-w5-cloudfeat → bot-W5-w1exec до 30.10 22:04 MSK |
| elenakonkova | `koz3-w2-cnx-c3sc-s3f4` | 9 | — | — | да | `HANDOFF/INFRA.md`: elenakonkova/koz3-w2-cnx-c3sc-s3f4 koz3-w2-cnx-c3sc-s3f4 Elena Konkova 2026-1 |
| elenakonkova | `koz3-w4-large-gate` | 9 | — | — | — | `HANDOFF/HANDOFF.md`: - large-gate v2: elenakonkova/koz3-w4-large-gate (re-pushed after fix ee038449; earlier Δ0 was missing-ae-OOF bug) |
| aaaa0000004 | `koz3-w4-v4chk` | 8 | — | — | — | `orch/COORD_LOG.md`: - 02.11 14:59 MSK / lease / acquire kernel:aaaa0000004/koz3-w4-v4chk → bot-W4-exec до 03.11 02:59 MSK v4 parity C/A/B/D/REF/NEG |
| aaaa0000004 | `koz3-w4-wheels` | 8 | — | — | — | `HANDOFF/INFRA.md`: aaaa0000004/koz3-w4-wheels koz3-w4-wheels 851466 202 |
| aaaadg4gg4g4g | `koz3-w3-geo5-l-s34` | 8 | — | — | — | `orch/COORD_LOG.md`: - 30.10 23:40 MSK / lease / set-kernel slot:aaaadg4gg4g4g/cpu → aaaadg4gg4g4g/koz3-w3-geo5-l-s34 by bot-W3-kurgan |
| aaaadg4gg4g4g | `koz3-w3-kurgan-cpu-seeds` | 8 | да | — | да | `HANDOFF/INFRA.md`: aaaadg4gg4g4g/koz3-w3-kurgan-cpu-seeds koz3-w3-kurgan-cpu-seeds AA AAdg4gg4g4g 20 |
| elenakonkova | `koz3-w2-maps` | 8 | — | — | — | `HANDOFF/INFRA.md`: elenakonkova/koz3-w2-maps koz3-w2-maps 2101891176 2026-10-30 09:0 |
| elenakonkova | `koz3-w3-kurgan-cpu-seeds` | 8 | да | — | да | `HANDOFF/INFRA.md`: elenakonkova/koz3-w3-kurgan-cpu-seeds koz3-w3-kurgan-cpu-seeds Elena Konkova 2026-1 |
| usertest1234 | `koz3-w3-geo5-l-s12` | 8 | — | — | — | `orch/COORD_LOG.md`: - 30.10 23:40 MSK / lease / set-kernel slot:usertest1234/cpu → usertest1234/koz3-w3-geo5-l-s12 by bot-W3-kurgan |
| usertest1234 | `koz3-w3-kurgan-cpu-seeds` | 8 | да | — | да | `HANDOFF/INFRA.md`: usertest1234/koz3-w3-kurgan-cpu-seeds koz3-w3-kurgan-cpu-seeds User Test1234 |
| elenakonkova | `koz3-dino-gor-f0` | 7 | — | — | — | `HANDOFF/INFRA.md`: elenakonkova/koz3-dino-gor-f0 koz3-dino-gor-f0 Elena Konkova 2026-1 |
| elenakonkova | `koz3-w4-li-post-gate` | 7 | — | — | — | `HANDOFF/INFRA.md`: elenakonkova/koz3-w4-li-post-gate koz3-w4-li-post-gate Elena Konkova 2026-1 |
| elenakonkova | `koz3-w5-cands` | 7 | — | — | — | `HANDOFF/INFRA.md`: elenakonkova/koz3-w5-cands koz3-w5-cands 1937990 2026-10-29 20:1 |
| elenakonkova | `koz3-w5-dino-emb` | 7 | — | — | да | `HANDOFF/INFRA.md`: elenakonkova/koz3-w5-dino-emb koz3-w5-dino-emb Elena Konkova 2026-1 |
| usertest1234 | `koz3-w3-kurgan-cpu-seeds-b` | 7 | — | — | — | `HANDOFF/INFRA.md`: usertest1234/koz3-w3-kurgan-cpu-seeds-b koz3-w3-kurgan-cpu-seeds-b User Test1234 |
| aaaa0000004 | `koz3-w4-dfine-ref103` | 6 | — | — | — | `HANDOFF/INFRA.md`: aaaa0000004/koz3-w4-dfine-ref103 koz3-w4-dfine-ref103 26358369 202 |
| elenakonkova | `koz3-li-aug-data` | 6 | — | — | — | `HANDOFF/INFRA.md`: elenakonkova/koz3-li-aug-data koz3-li-aug-data 210485079 2026-11-01 08:4 |
| elenakonkova | `koz3-sp-large-f0` | 6 | да | — | — | `kaggle/2026-10-31/sp_large_f0/kernel-metadata.json`: {"id": "elenakonkova/koz3-sp-large-f0", "title": "koz3-sp-large-f0", "code_file": "koz3-sp-large-f0.py", |
| elenakonkova | `koz3-sp-large-f0-data` | 6 | да | — | — | `HANDOFF/INFRA.md`: elenakonkova/koz3-sp-large-f0-data koz3-sp-large-f0-data 2192843820 2026-10-31 13:2 |
| usertest1234 | `koz3-w3-kurgan-cpu-data` | 6 | да | — | — | `HANDOFF/INFRA.md`: usertest1234/koz3-w3-kurgan-cpu-data koz3-w3-kurgan-cpu-data 201105613 20 |
| aaaadg4gg4g4g | `koz3-w3-kurgan-cpu-data` | 5 | да | — | — | `HANDOFF/INFRA.md`: aaaadg4gg4g4g/koz3-w3-kurgan-cpu-data koz3-w3-kurgan-cpu-data 201105613 2 |
| aaaadg4gg4g4g | `koz3-w3-kurgan-cpu-seeds-b` | 5 | — | — | — | `HANDOFF/INFRA.md`: aaaadg4gg4g4g/koz3-w3-kurgan-cpu-seeds-b koz3-w3-kurgan-cpu-seeds-b AA AAdg4gg4g4g 20 |
| elenakonkova | `koz3-peak-rerank-f0` | 5 | — | — | — | `HANDOFF/INFRA.md`: elenakonkova/koz3-peak-rerank-f0 koz3-peak-rerank-f0 Elena Konkova 2026-1 |
| aaaa0000004 | `koz3-w3-geo5-feat-base` | 4 | да | — | — | `HANDOFF/INFRA.md`: aaaa0000004/koz3-w3-geo5-feat-base koz3-w3-geo5-feat-base 574042221 202 |
| aaaa0000004 | `koz3-w3-geo5-feat-n40r8` | 4 | — | — | — | `HANDOFF/INFRA.md`: aaaa0000004/koz3-w3-geo5-feat-n40r8 koz3-w3-geo5-feat-n40r8 2315162646 202 |
| aaaa0000004 | `koz3-w3-kurgan-cpu-data` | 4 | да | — | — | `HANDOFF/INFRA.md`: aaaa0000004/koz3-w3-kurgan-cpu-data koz3-w3-kurgan-cpu-data 201105613 202 |
| aaaa0000004 | `koz3-w4-v3amin-chk` | 4 | — | — | — | `HANDOFF/INFRA.md`: aaaa0000004/koz3-w4-v3amin-chk koz3-w4-v3amin-chk AA AA0000004 2026-1 |
| aaaadg4gg4g4g | `koz3-w5-cloudfeat-in` | 4 | — | — | — | `orch/COORD_LOG.md`: - 30.10 08:30 MSK / lease / acquire dataset:aaaadg4gg4g4g/koz3-w5-cloudfeat-in → bot-W5-w1exec до 30.10 12:30 MSK CI2-1 input; executor bot- |
| elenakonkova | `koz3-dino-gor-data` | 4 | — | — | — | `HANDOFF/INFRA.md`: elenakonkova/koz3-dino-gor-data koz3-dino-gor-data 331681446 2026-10-31 18:5 |
| elenakonkova | `koz3-dino-gor-oof5` | 4 | — | — | — | `HANDOFF/INFRA.md`: elenakonkova/koz3-dino-gor-oof5 koz3-dino-gor-oof5 Elena Konkova 2026-1 |
| elenakonkova | `koz3-w4-li-post-full` | 4 | — | — | — | `HANDOFF/INFRA.md`: elenakonkova/koz3-w4-li-post-full koz3-w4-li-post-full Elena Konkova 2026-1 |
| aaaa0000004 | `koz3-w3-geo5-n40r8-s12` | 3 | — | — | — | `orch/COORD_LOG.md`: - 31.10 00:40 MSK / lease / acquire kernel:aaaa0000004/koz3-w3-geo5-n40r8-s12 → bot-W3-kurgan до 31.10 10:40 MSK geo5 OOF n40r8 (63lv/400r m |
| aaaa0000004 | `koz3-w4-v3a-min-check` | 3 | — | — | — | `HANDOFF/INFRA.md`: aaaa0000004/koz3-w4-v3a-min-check koz3-w4-v3a-min-check 526506401 202 |
| aaaa0000004 | `koz3-w4-v3on-check` | 3 | — | — | — | `HANDOFF/INFRA.md`: aaaa0000004/koz3-w4-v3on-check koz3-w4-v3on-check 661130738 202 |
| aaaa0000004 | `koz3-w4-wheels-v3` | 3 | — | — | — | `HANDOFF/INFRA.md`: aaaa0000004/koz3-w4-wheels-v3 koz3-w4-wheels-v3 27005800 202 |
| aaaadg4gg4g4g | `koz3-w3-geo5-feat-n40r8` | 3 | — | — | — | `HANDOFF/INFRA.md`: aaaadg4gg4g4g/koz3-w3-geo5-feat-n40r8 koz3-w3-geo5-feat-n40r8 2315162646 2 |
| elenakonkova | `koz3-cnx-li-c3sc-` | 3 | — | — | да | `HANDOFF_W2.md`: / Веса c3-sc `weights_cnx_c3-sc_f{0..4}.pt` для s0, s1, s2 (15 шт.), `ckpt_*` s2 f2–f4, li_hm `ckpt_cnx_c3-sc_f{0..4}.pt` (у li_hm только ck |
| elenakonkova | `koz3-peak-rerank-peaks` | 3 | — | — | — | `HANDOFF/INFRA.md`: elenakonkova/koz3-peak-rerank-peaks koz3-peak-rerank-peaks 2741563 2026-10-31 19:2 |
| elenakonkova | `koz3-s3-index` | 3 | да | — | — | `HANDOFF/INFRA.md`: elenakonkova/koz3-s3-index koz3-s3-index Elena Konkova 2026-1 |
| elenakonkova | `koz3-w1-sat2-gate` | 3 | — | — | да | `HANDOFF/INFRA.md`: elenakonkova/koz3-w1-sat2-gate koz3-w1-sat2-gate Elena Konkova 2026-1 |
| elenakonkova | `koz3-w2-hmcp-c3sc-f01234` | 3 | — | — | — | `orch/bot-W2/PREREG_W2_cnx.md`: - Вариант (единственный, без перебора): ядро `elenakonkova/koz3-w2-hmcp-c3sc-f01234` = P-W2-7 lihm (U-Net tu-convnext_tiny, scratch, c3-sc,  |
| elenakonkova | `koz3-w3-geo5-feat-n40r8` | 3 | — | — | — | `HANDOFF/INFRA.md`: elenakonkova/koz3-w3-geo5-feat-n40r8 koz3-w3-geo5-feat-n40r8 2315162646 2026-10-30 20:5 |
| elenakonkova | `koz3-w3-kurgan-cpu-data` | 3 | да | — | — | `HANDOFF/INFRA.md`: elenakonkova/koz3-w3-kurgan-cpu-data koz3-w3-kurgan-cpu-data 201105613 2026-10-30 08:2 |
| usertest1234 | `koz3-w3-geo5-feat-n40r8` | 3 | — | — | — | `HANDOFF/INFRA.md`: usertest1234/koz3-w3-geo5-feat-n40r8 koz3-w3-geo5-feat-n40r8 2315162646 20 |
| aaaa0000004 | `aaaadg` | 2 | — | — | да | `orch/bot-W3/STATUS_W3.md`: 23:20 MSK / W3 / coord перезапустил 3× seeds (usertest/aaaa0000004/aaaadg) с ТЕМ ЖЕ кодом (kernels pull = идентичен) — не трогаю; v5 в QUEUE |
| aaaa0000004 | `koz3-li-tin-59-t128-bundle` | 2 | да | — | — | `HANDOFF/INFRA.md`: aaaa0000004/koz3-li-tin-59-t128-bundle KOZ3 Li TIN 59 T128 Bundle Node 2 291348379 202 |
| aaaa0000004 | `koz3-li-tin-59-t128-dual-t4` | 2 | да | — | — | `HANDOFF/INFRA.md`: aaaa0000004/koz3-li-tin-59-t128-dual-t4 koz3-li-tin-59-t128-dual-t4 AA AA0000004 2026-1 |
| aaaa0000004 | `koz3-w3-kurgan-cpu-seeds.py` | 2 | — | — | да | `orch/BACKUP_2026-11-02.md`: / cpuk/k_aaaa0000004/koz3-w3-kurgan-cpu-seeds.py / 30005 / `06a35a22133cfc90d22a90ef9699e3921a907c9882861e022091c238b7b9b20d` / / |
| aaaa0000004 | `koz3-w4-v3onfix` | 2 | — | — | да | `orch/bot-W4/STATUS.md`: 01.11 23:45 MSK / W4 / Kaggle v5 koz3-w4-v3onchk (архив v3on_b447649 sha 1406c01a…ef41d) COMPLETE: C_default PASS (FAIL 0, WARN 3), A_regex2 |
| aaaadg4gg4g4g | `koz3-li-tin-59-t128-bundle` | 2 | да | — | — | `HANDOFF/INFRA.md`: aaaadg4gg4g4g/koz3-li-tin-59-t128-bundle KOZ3 Li TIN 59 T128 Bundle Node 3 291348379 2 |
| aaaadg4gg4g4g | `koz3-li-tin-59-t128-dual-t4` | 2 | да | — | — | `HANDOFF/INFRA.md`: aaaadg4gg4g4g/koz3-li-tin-59-t128-dual-t4 koz3-li-tin-59-t128-dual-t4 AA AAdg4gg4g4g 20 |
| aaaadg4gg4g4g | `koz3-w3-kurgan-cpu-seeds.py` | 2 | — | — | да | `orch/BACKUP_2026-11-02.md`: / cpuk/k_aaaadg4gg4g4g/koz3-w3-kurgan-cpu-seeds.py / 30005 / `751451d2e76640c9ed40b22468b3f3d983563a0285e9e049b2ec3658dad67862` / / |
| elenakonkova | `koz3-gt-train-v2` | 2 | — | — | — | `docs/PREREG_W1_SATDET.md`: elenakonkova/koz3-gt-train-v2: GT, markup sp_or polygons only). Rasters warped to UTM 0.5 m. No S3, no credentials. |
| elenakonkova | `koz3-w1-sat2-src` | 2 | — | — | — | `orch/bot-W1/STATUS.md`: 01.11 20:08 MSK / bot-W1-aeor / sat2-gate v1 ERROR 0 с: script-ядро несёт один файл, vendor/data/w5_rerank/scripts не было. Фикс: приватный  |
| elenakonkova | `koz3-w2-relief` | 2 | — | — | — | `orch/COORD_LOG.md`: - 30.10 11:23 MSK / lease / acquire dataset:elenakonkova/koz3-w2-relief → bot-W2-lidar до 30.10 19:23 MSK W2 backup for handoff (Elena t16u) |
| elenakonkova | `koz3-w3-bigobj-adacrop` | 2 | — | — | — | `orch/bot-W3/PREP_CI2_5.md`: # A выдаёт: orch/lease.sh acquire slot:acc1/gpu-<n> bot-W3-bigobj <ttl> --kernel elenakonkova/koz3-w3-bigobj-adacrop --note "CI2-5" |
| elenakonkova | `koz3-w3-geo5-l-s56` | 2 | — | — | — | `orch/COORD_LOG.md`: - 30.10 23:49 MSK / lease / acquire kernel:elenakonkova/koz3-w3-geo5-l-s56 → bot-W3-kurgan до 31.10 09:49 MSK geo5 OOF L s5-6; MagGr E1_G1 f |
| elenakonkova | `koz3-w5-cloudfeat-in` | 2 | — | — | — | `orch/COORD_LOG.md`: - 30.10 11:12 MSK / coord / 15-мин проверка: box load 12.4/8 (полно). acc1/cpu простаивал с 10:42 → push elenakonkova/koz3-w5-cloudfeat (CI2 |
| elenakonkova | `koz3-wsr-data` | 2 | — | — | — | `HANDOFF/INFRA.md`: elenakonkova/koz3-wsr-data koz3-wsr-data 54794887 2026-10-31 20:5 |
| usertest1234 | `koz3-li-tin-59-t128-bundle` | 2 | да | — | — | `HANDOFF/INFRA.md`: usertest1234/koz3-li-tin-59-t128-bundle KOZ3 Li TIN 59 T128 Bundle Node 1 291348379 20 |
| usertest1234 | `koz3-li-tin-59-t128-dual-t4` | 2 | да | — | — | `HANDOFF/INFRA.md`: usertest1234/koz3-li-tin-59-t128-dual-t4 koz3-li-tin-59-t128-dual-t4 User Test1234 |
| usertest1234 | `koz3-w3-kurgan-cpu-seeds.py` | 2 | — | — | да | `orch/BACKUP_2026-11-02.md`: / cpuk/k_usertest1234/koz3-w3-kurgan-cpu-seeds.py / 30005 / `6d13566f3d868500d73b7a83478d3ab68f3ac584548aa3828e16354592bacd4d` / / |
| usertest1234 | `koz3-w4-fr1h-check` | 2 | — | — | — | `HANDOFF/INFRA.md`: usertest1234/koz3-w4-fr1h-check koz3-w4-fr1h-check User Test1234 |
| usertest1234 | `koz3-w4-pixclean-cache` | 2 | — | — | — | `orch/bot-W4/kaggle/koz3-w4-pix-clean-fit/kernel-metadata.json`: "dataset_sources": ["usertest1234/koz3-w4-pixclean-cache"], |
| aaaa0000004 | `koz3-gpu-probe` | 1 | да | — | — | `HANDOFF/INFRA.md`: aaaa0000004/koz3-gpu-probe koz3-gpu-probe AA AA0000004 2026-1 |
| aaaa0000004 | `koz3-gt-train-v2` | 1 | — | — | — | `HANDOFF/INFRA.md`: aaaa0000004/koz3-gt-train-v2 koz3-gt-train-v2 11465528 202 |
| aaaa0000004 | `koz3-w4-fr1h-geo` | 1 | — | — | — | `HANDOFF/INFRA.md`: aaaa0000004/koz3-w4-fr1h-geo koz3-w4-fr1h-geo AA AA0000004 2026-1 |
| aaaadg4gg4g4g | `koz3-gpu-probe` | 1 | да | — | — | `HANDOFF/INFRA.md`: aaaadg4gg4g4g/koz3-gpu-probe koz3-gpu-probe AA AAdg4gg4g4g 20 |
| aaaadg4gg4g4g | `koz3-gt-train-v2` | 1 | — | — | — | `HANDOFF/INFRA.md`: aaaadg4gg4g4g/koz3-gt-train-v2 koz3-gt-train-v2 11465528 2 |
| aaaadg4gg4g4g | `koz3-w4-fr1h-gate` | 1 | — | — | — | `HANDOFF/INFRA.md`: aaaadg4gg4g4g/koz3-w4-fr1h-gate koz3-w4-fr1h-gate AA AAdg4gg4g4g 20 |
| elenakonkova | `` | 1 | — | — | да | `orch/TPU_HEDGE_ACCESS_20261030.md`: Hedge kernel copy must use `id: dmitrijkv/koz3-w5-dino-vitb-tpu` (current metadata id is elenakonkova/...). Re-check acc1 vitb status: if it |
| elenakonkova | `koz3-acc2-gpu-smoke` | 1 | — | — | да | `HANDOFF/INFRA.md`: elenakonkova/koz3-acc2-gpu-smoke koz3-acc2-gpu-smoke Elena Konkova 2026-1 |
| elenakonkova | `koz3-bot7-geo-gate` | 1 | да | — | — | `HANDOFF/INFRA.md`: elenakonkova/koz3-bot7-geo-gate koz3-bot7-geo-gate Elena Konkova 2026-1 |
| elenakonkova | `koz3-bot7-sat-recall` | 1 | да | — | — | `HANDOFF/INFRA.md`: elenakonkova/koz3-bot7-sat-recall koz3-bot7-sat-recall Elena Konkova 2026-1 |
| elenakonkova | `koz3-bot8-oof-honest10` | 1 | — | — | да | `HANDOFF/INFRA.md`: elenakonkova/koz3-bot8-oof-honest10 koz3-bot8-oof-honest10 Elena Konkova 2026-1 |
| elenakonkova | `koz3-bot8-snap` | 1 | — | — | — | `HANDOFF/INFRA.md`: elenakonkova/koz3-bot8-snap koz3-bot8-snap 479642150 2026-10-29 17:0 |
| elenakonkova | `koz3-e2e-baseline-v2` | 1 | — | — | — | `reports/bot6_pixgbm.md`: - Existing `elenakonkova/koz3-e2e-baseline-v2` noted but 403 from dmitrijkv token → tip REF staged locally |
| elenakonkova | `koz3-e2e-tools` | 1 | — | — | — | `HANDOFF/INFRA.md`: elenakonkova/koz3-e2e-tools koz3-e2e-tools 32879 2026-10-29 17:3 |
| elenakonkova | `koz3-w1-ae-dfine-gate` | 1 | да | — | — | `reports/h2h_20261101/RESULTS.md`: / elenakonkova/koz3-w1-ae-dfine-gate (rerun 21:22 MSK) / ERROR (script FAILED, flag_dead) / 1-region smoke (015_Демидовка): OFF/A+B/A+B+C al |
| elenakonkova | `koz3-w1-x` | 1 | — | — | — | `orch/PROTOCOL.md`: orch/lease.sh acquire slot:acc1/gpu-1 bot-W1-aeor 8 --kernel elenakonkova/koz3-w1-x --note "..." |
| elenakonkova | `koz3-w2-` | 1 | — | — | да | `HANDOFF_W2.md`: - Стейджинг бэкапа: `$K/w2/bk/stage/{oof,weights,maps,gate}` (хардлинки + dataset-metadata.json). Новая версия: `cd $K/w2/bk/stage/<s> && $P |
| elenakonkova | `koz3-w2-cnx-c5sc-f01` | 1 | — | — | — | `w2_kernels/elenakonkova/koz3-w2-cnx-c5sc-f01/kernel-metadata.json`: "id": "elenakonkova/koz3-w2-cnx-c5sc-f01", |
| elenakonkova | `koz3-w2-cnx-c5sc-f23` | 1 | — | — | — | `w2_kernels/elenakonkova/koz3-w2-cnx-c5sc-f23/kernel-metadata.json`: "id": "elenakonkova/koz3-w2-cnx-c5sc-f23", |
| elenakonkova | `koz3-w2-cnx-c5sc-f4` | 1 | — | — | — | `w2_kernels/elenakonkova/koz3-w2-cnx-c5sc-f4/kernel-metadata.json`: "id": "elenakonkova/koz3-w2-cnx-c5sc-f4", |
| elenakonkova | `koz3-w3-geo5-feat-base` | 1 | да | — | — | `HANDOFF/INFRA.md`: elenakonkova/koz3-w3-geo5-feat-base koz3-w3-geo5-feat-base 574042221 2026-11-01 17:1 |
| usertest1234 | `hod-mysli-2026-tag-gpu-worker` | 1 | да | — | — | `HANDOFF/INFRA.md`: usertest1234/hod-mysli-2026-tag-gpu-worker hod-mysli-2026-tag-gpu-worker User Test1234 |
| usertest1234 | `koz3-gpu-probe` | 1 | да | — | — | `HANDOFF/INFRA.md`: usertest1234/koz3-gpu-probe koz3-gpu-probe User Test1234 |
| usertest1234 | `koz3-gt-train-v2` | 1 | — | — | — | `HANDOFF/INFRA.md`: usertest1234/koz3-gt-train-v2 koz3-gt-train-v2 11465528 20 |
| usertest1234 | `koz3-w3-geo5-feat-base` | 1 | да | — | — | `HANDOFF/INFRA.md`: usertest1234/koz3-w3-geo5-feat-base koz3-w3-geo5-feat-base 574042221 20 |
| usertest1234 | `koz3-w4-pix-clean-fit` | 1 | — | — | — | `orch/bot-W4/kaggle/koz3-w4-pix-clean-fit/kernel-metadata.json`: "id": "usertest1234/koz3-w4-pix-clean-fit", |
