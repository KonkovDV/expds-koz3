# Провенанс весов `.pt` в git (15 файлов) — 29.10.2026

Зачем: правило конкурса — **нельзя данные прошлого этапа** (КОЗ №2 «Сканирование», Kaggle `ackrome/satelite-scan`);
можно предобученные веса COCO / ImageNet / DINOv2 / Objects365 / ADAF. Составлено bot-1 (executor) по `main` 164707a,
только по содержимому репозитория (коммиты, `notes/`, `docs/`, метаданные внутри чекпойнтов). Обучающие кэши
(`F:/expds3/work/cache/*`) лежат на ПК владельца и в git не входят — их здесь проверить нельзя (см. «Что осталось»).

## Итог

* **FORBIDDEN (данные КОЗ №2): 0.** **UNKNOWN: 0.** Все 15 — свои U-Net/ResNet-18 из `expds/train.py`
  (или `expds/crop_model.py` для `sp_crop.pt`), обучены на трейне КОЗ-3 (`F:/expds3/train-expds-3`, семьи/участки
  КОЗ-3: «Боровичи», «Эрентуна», «Курганы», «Гнёздово», нумерация `NNN_Имя`), энкодер инициализирован весами
  ImageNet (разрешено; `docs/THIRD_PARTY.md`), `sp_crop.pt` — без предобучения (`pretrained: False`).
* Уровень доказательства: 9 файлов — **документировано** (команда обучения / лог / заметка с числом тайлов);
  6 — **косвенно** (запись в `AI_DECLARATION.md`/планах + метаданные чекпойнта, лога финального обучения в git нет).
  Косвенные помечены, но к UNKNOWN не отнесены: см. аргумент по датам ниже.
* **Аргумент по датам (сильный):** каждый `.pt` добавлен одним коммитом 21.10–28.10 и **больше не менялся**
  (`git log -- models/X.pt` = 1 коммит). Датасет «Сканирование» впервые всплыл в проекте **29.10 08:01 МСК**
  (бот 9 читал метаданные и скачал малую часть geojson/UTM.json; ничего не обучалось; удалено ~08:20) —
  `CHANGELOG.md` 29.10, `docs/bots/2026-10-29/bot-9/DATA_PROTOCOL_SCAN_NOTE.md`. То есть все веса старше
  первого касания запрещённых данных.
* В конфиге (`configs/default.toml`) стоят 10 из 15; 5 ранних (`ae_r18`, `li_r18`, `li_r18_all`, `li_r18_hard`,
  `spor_arch_r18`) в конфиге нет и в архив сдачи не попадают (`scripts/build_submission.py` берёт только из конфига).

## Таблица

Автор коммитов — `expds-local` (локальная машина владельца). Время коммита — МСК (+0300).

| Файл | Размер, Б | sha256 | Введён (коммит, автор, дата) | В конфиге | Инициализация | Данные обучения (что известно) | Источник в репо | Доказательство | Класс |
|---|---|---|---|---|---|---|---|---|---|
| `models/ae_fort_mask_r18.pt` | 57426479 | `36d64e1a8a14d850dbf1494a11629e2a4ff7cea218b10dd462c5b827cf58bba7` | `bd05936` expds-local, 22.10 21:42 | да | ResNet-18 ImageNet (smp/torchvision) | markup ae, класс fortifikatsii; 20 эпох на 4327 тайлах, кэш `F:/expds3/work/cache/ae_fort_all` (шарды `ae_all_r05_t128_s04`) | `notes/ae_fort_mask_2026-10-22.json`, `docs/AGENT_PLAN.md:152`, `docs/AI_DECLARATION.md` | документировано | **OK: трейн КОЗ-3 + ImageNet-инициализация** |
| `models/ae_r18.pt` | 57421591 | `83b4cc32b17f5c17604ffee530f4eec23f37b5017449390c6a70f80ea43cca98` | `454bc02` expds-local, 21.10 10:59 | нет | ResNet-18 ImageNet (smp/torchvision) | markup ae, 5 классов; `expds.train --cache F:/expds3/work/cache/ae_all_r05_t128_s04 --val-families Боровичи`; лог 4082/245 тайлов | `notes/gpu_tasks.json` (ae_final), `notes/ae_final.log` | документировано | **OK: трейн КОЗ-3 + ImageNet-инициализация** |
| `models/li_dev_r18.pt` | 57426119 | `2b655c809cb89c9394fb36ae9fec1d710b1d17d35a671834bdd80404b28265ab` | `c616c19` expds-local, 24.10 08:52 | да | ResNet-18 ImageNet (smp/torchvision) | markup li, каналы dev_s/l/xl из LAS трейна; 5 фолдов + финал (пакет «окно 256 м и канал σ», 0,353793) | `docs/AI_DECLARATION.md` (23.10, «Канал σ лидара»), `docs/AGENT_PLAN.md:69`, метаданные чекпойнта | косвенно (нет лога финального обучения) | **OK: трейн КОЗ-3 + ImageNet-инициализация** |
| `models/li_fort_mask_r18.pt` | 57426607 | `eb11cff8be474501d26fb6755268150abeaed42289512a6dd507b0571d4934e9` | `c616c19` expds-local, 24.10 08:52 | да | ResNet-18 ImageNet (smp/torchvision) | markup li, класс fortifikatsii, окно 384 м, LAS трейна | `docs/AI_DECLARATION.md:54` (23.10), `docs/AGENT_PLAN.md:21`, метаданные чекпойнта | косвенно (нет лога финального обучения) | **OK: трейн КОЗ-3 + ImageNet-инициализация** |
| `models/li_gor_mask_r18.pt` | 57426355 | `39702cd8a5d92d0c0c06132690febc265519ed7a35b3c6ff02ed1fd7c1fdd8ef` | `f274217` expds-local, 22.10 00:20 | да | ResNet-18 ImageNet (smp/torchvision) | markup li, класс gorodishcha; 21 эпоха на всех 486 тайлах (медиана фолдов) | `notes/li_gor_mask_2026-10-21.json:33`, `docs/PLAN.md:171`, `docs/AI_DECLARATION.md:37` | документировано | **OK: трейн КОЗ-3 + ImageNet-инициализация** |
| `models/li_r18.pt` | 57421591 | `c36149aba08bc619bdae67169fb51437bf4d08776f6f524b46d2a58b68ba90b5` | `454bc02` expds-local, 21.10 10:59 | нет | ResNet-18 ImageNet (smp/torchvision) | markup li, 5 классов; ранняя модель (пример команды в `expds/train.py:8`, `--val-families Гнёздово`) | `expds/train.py:8`, `models/README.md` | косвенно | **OK: трейн КОЗ-3 + ImageNet-инициализация** |
| `models/li_r18_all.pt` | 57427719 | `2ed166cd9b37c3514438b885ec196cd6c54aeba9331bb5a4e81c55faa71ef531` | `454bc02` expds-local, 21.10 10:59 | нет | ResNet-18 ImageNet (smp/torchvision) | markup li; `expds.train --cache F:/expds3/work/cache/li_tin_all51_t128 --val-families Эрентуна`; кэш из `expds.tiles --root F:/expds3/train-expds-3`; лог 3429/140 тайлов | `notes/gpu_tasks.json` (li_final_all51), `notes/cpu_tasks.json` (li_all51_cache), `notes/li_final_all.log` | документировано | **OK: трейн КОЗ-3 + ImageNet-инициализация** |
| `models/li_r18_hard.pt` | 57427907 | `100bcc21afe8fc09be5362e19e2621ec815db28a38cad82a6bd9a5dffa7cc7d9` | `454bc02` expds-local, 21.10 10:59 | нет | ResNet-18 ImageNet (smp/torchvision) | markup li; трудные негативы на кэше трейна (hard_oof) | `notes/hard_oof.log`, `notes/hardneg_mine.log` | косвенно | **OK: трейн КОЗ-3 + ImageNet-инициализация** |
| `models/li_r18_hard2.pt` | 57428095 | `0311e877e148e0ec3acb6f3b13a7fb12affe5e0b58a60d3524a2df956b3793b1` | `454bc02` expds-local, 21.10 10:59 | да | ResNet-18 ImageNet (smp/torchvision) | markup li, курганы; трудные негативы v2 на кэше трейна (`li_tin_hard2_t128`) | `notes/li_hard2_final.log` (4083/177 тайлов), `notes/hardneg2_mine.log`, `notes/hard2_oof.log`, `docs/AGENT_PLAN.md:234`, `analysis/2026-10-28/oof59_pc/` | документировано | **OK: трейн КОЗ-3 + ImageNet-инициализация** |
| `models/sp_crop.pt` | 44783371 | `4a41b0caada94d387721eb13526071469d5c70504b9dfb6743e7923a1f39cbeb` | `931be93` expds-local, 28.10 04:33 | да | нет (с нуля) | кроп-классификатор resnet18, `pretrained: False` в чекпойнте; 1239 позитивов / 56587 кропов кандидатов стека трейна | `docs/THIRD_PARTY.md:63` («только трейн КОЗ-3»), `docs/PLAN.md:9,33`, метаданные чекпойнта | документировано | **OK: трейн КОЗ-3 + ImageNet-инициализация** (без предобучения) |
| `models/spor_arch_r18.pt` | 57425979 | `9478716f2b41567a25733919a92d55f2f199c8ca1d09c337ac2571705bb30cac` | `c616c19` expds-local, 24.10 08:52 | нет | ResNet-18 ImageNet (smp/torchvision) | markup sp_or, класс arkhitektury, круги 2 м | `docs/bots/2026-10-28/bot-5/results/analysis/ckpt_meta.json`, `docs/bots/2026-10-28/bot-5/NOLIDAR.md` | косвенно | **OK: трейн КОЗ-3 + ImageNet-инициализация** |
| `models/spor_focal_r18.pt` | 57426743 | `88bbd1a24bf4bd4a2dae8c7855a95767e117124955273c9df1b2255c96fdd88b` | `dd438e7` expds-local, 21.10 12:37 | да | ResNet-18 ImageNet (smp/torchvision) | markup sp_or, курганы, карта центров (фокальная); веса на всех тайлах трейна | `docs/AI_DECLARATION.md:20` (21.10), `docs/PLAN.md:240,264` | документировано | **OK: трейн КОЗ-3 + ImageNet-инициализация** |
| `models/spor_focal_wide_r18.pt` | 57427747 | `aa09855cb95babc8bfaf8018e6d32763ff0fd8356db5c4fb41fa31333a4f742c` | `c616c19` expds-local, 24.10 08:52 | да | ResNet-18 ImageNet (smp/torchvision) | markup sp_or, курганы, окно 256 м, карта центров | `docs/AI_DECLARATION.md:51-52` (22.10), `ckpt_meta.json` | косвенно (нет лога финального обучения) | **OK: трейн КОЗ-3 + ImageNet-инициализация** |
| `models/spor_gor_r18.pt` | 57425599 | `00d5c08539d1bced03d373c018d9e31432d987c7d848196a4afac6b0d44f1cbe` | `454bc02` expds-local, 21.10 10:59 | да | ResNet-18 ImageNet (smp/torchvision) | markup sp_or, класс gorodishcha; `expds.train --cache F:/expds3/work/cache/spor_gor_r2_t128 --val-families Курганы` | `notes/gpu_tasks.json` (spor_gor_final), `ckpt_meta.json` | документировано | **OK: трейн КОЗ-3 + ImageNet-инициализация** |
| `models/surf_arch_r18.pt` | 57426107 | `8266add934346e5c675cda0c4aeb84ee39be3652e9308136a789673d768011a7` | `38f0986` expds-local, 27.10 23:56 | да | ResNet-18 ImageNet (smp/torchvision) | markup or+sp_or, класс arkhitektury, Unet 256; кэш `surf_arch_r05_t256` (ортофото 819 + спутник 2082 тайлов трейна) | `docs/AI_DECLARATION.md:146,149` (27.10), `docs/LB_LOG.md`, `ckpt_meta.json` | документировано | **OK: трейн КОЗ-3 + ImageNet-инициализация** |

Метаданные внутри чекпойнтов (ключи `markup`, `classes`, `li_source`, `li_tags`, `epoch`, `val_loss`) прочитаны без
загрузки тензоров; сводка — `docs/bots/2026-10-28/bot-5/results/analysis/ckpt_meta.json`. Ссылок на внешние
наборы в них нет. `markup` (`li`/`ae`/`sp_or`/`or`) — слои разметки КОЗ-3.

## Поиск по репозиторию запрещённых данных

`git grep -n -i -E 'satelite-scan|ackrome|Сканирование|koz2|KOZ #2|koz 2'` по дереву `main` (без `*.pt`):

* **Код, который читает «Сканирование» / `ackrome/satelite-scan`: не найден.** Хуков/флагов для этих данных нет.
* Все упоминания — запреты и протоколы: `README.md:16`, `CHANGELOG.md` (29.10), `FOR_BOT9_GORLOC.md:40`,
  `docs/STATUS_2026-10-29.md:29`, `docs/AUDIT_CLAUDE_2026-10-29.md`, `docs/bots/2026-10-29/bot-{5,9,10}/*`
  (`PREREG_scan.md` — отменён, `REPORT10_accept.md` — «снова ЗАПРЕЩЁН»), `scripts/gor_dense/dense_probe.py:18`.
* Kaggle-сборщики `kaggle/2026-10-29/bot5/{gorloc,dinov2reg}/build.py` **проверяют**, что в ядре нет
  `satelite-scan`/`ackrome` (`assert`); в ядре dinov2reg — комментарий «запрещены — не используются».
* OSINT и разбор конкурентов (`docs/research/*`, `docs/bots/2026-10-28/bot-3/MATANERDY_VERDICT.md`,
  `docs/bots/2026-10-28/bot-8/NOTES_*KOZ2*.md`, `analysis/2026-10-28/bots/bot-3/mata_sanity/koz3_rules_sim.py`)
  — описания чужих решений КОЗ №2 и их метрики; данных КОЗ №2 не содержат и в обучение не идут.
* «сканирование» в `docs/background/tr_*.txt`, `notes/tr_*.txt` — термин ТР («лазерное сканирование»), не датасет.
* Совпадение `KoZ2` в `patches/2026-10-28/bot-8/bot8-rust-metric.diff:6401` — байты base85 бинарного патча.

## Что осталось (не закрыто этим документом)

1. Кэши обучения на ПК (`F:/expds3/work/cache/<кэш>/meta.json`) — список участков. Сверить с перечнем участков
   трейна КОЗ-3 для 6 «косвенных» файлов: `li_dev_r18`, `li_fort_mask_r18`, `li_r18`, `li_r18_hard`,
   `spor_arch_r18`, `spor_focal_wide_r18`. Это 5 минут на ПК владельца; до этого — «OK по датам и записям».
2. «34 новых LAS» (`notes/fetch_las.log`, `notes/las_todo.txt`) — дозагрузка крупных облаков **того же трейна
   КОЗ-3** (`train-expds-3/NNN_Имя/01_*_La_точки/*.las`), 21.10; к «Сканированию» отношения не имеет.
3. Веса не в git (LightGBM `models/pixgbm/all/*.txt`, `rerank_arch_lgbm`, ConvNeXt bot-9, D-FINE, DINOv2-голова)
   здесь не разбирались — см. `docs/THIRD_PARTY.md` («только трейн КОЗ-3») и `models/pixgbm/all/MODELS_NOT_IN_GIT.md`.
