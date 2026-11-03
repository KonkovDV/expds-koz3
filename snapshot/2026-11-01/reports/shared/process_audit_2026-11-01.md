# Аудит процесса KOZ3 — 01.11.2026 ~19:22 MSK (Europe/Moscow)

Проект: `KonkovDV/expds-koz3` (КОЗ №3). Дедлайн **11.11.2026**.  
Аудитор: executor (coord-deep). Режим: без fan-out / cloud agents; GPU slot-2 и пакет `ae-dfine-gate` не трогались.

## 0. Краткий итог действий

| Действие | Результат |
|---|---|
| Push CPU пакетов | **Не выполнено** (см. §1) — ни один готовый пакет не прошёл offline-CPU + dataset visibility; к 19:21 все 3 CPU-акка уже арендованы bot-W4-gate под FR1h |
| Dry-run `tta-gate` | OK на боксе (`/tmp/tta-gate-dry/`, dry=1) |
| `elenakonkova/koz3-w2-oof` → fam0 / `family_clusters.json` | **НЕТ** (для bot4) |
| Отчёт | этот файл |
| COORD_LOG | краткая запись добавлена |

---

## 1. Заполнение idle CPU — что пушили / почему нет

### 1.1 Живые слоты на момент аудита (~19:15→19:22)

| Акк | Факт (kernels status) | Примечание |
|---|---|---|
| elenakonkova | `koz3-lidar-aug-f0` **RUNNING** (folds 3,4) | 2-й GPU не трогать (ae-dfine-gate готовится другим) |
| dmitrijkv | `koz3-w1-lidar-aug-f2` **RUNNING** (ETA ~21:45) | то же |
| usertest1234 / aaaadg4gg4g4g / aaaa0000004 | до 19:21 idle → **с 19:21 аренда bot-W4-gate** (`koz3-w4-fr1h-{check,gate,geo}`) | столкновения избегать |

### 1.2 Пакеты `/workspace/kaggle/push/`

| Пакет | Вердикт push | Почему |
|---|---|---|
| **pixgbm-cr** | **НЕ пушить сейчас** | В metadata нет REF/CAND (нужны zip v2 `eb268d6` + clean-pack / `models_pix_clean`); `enable_internet=true`; geo-датасеты `dmitrijkv/koz3-w4-geo-s000/s001` не видны CPU-аккам как owned; без аттачей ядро уйдёт в exit 3. Ветка `bots/bot6-pixgbm-cleanroom` @`37240147` (не использовать PR#20 / `9b411c4` как clean-room ответ). SIGTERM hang к clean-пути не относится (threads). **Gate output на Kaggle ещё нет** — только orch P1 parity на Drive (`out/geo-clean-pack/…`, sticky 01.11). |
| **tta-gate** | **НЕ пушить до Ae gate** | Slug в metadata исправлен: `dmitrijkv/koz3-tta-gate` (не `koz3-ae-dfine-gate`). Но: `enable_gpu=true`, internet, `koz3-s3-cred`, нужен Ae ckpt для `tta_ae`. Варианты `bprime`/`tta_ms_sat` в DRY на CPU работают, **реальный** прогон без S3/весов/честных данных на CPU-no-internet невозможен. `lease.sh` не принимает роль `bot-8` (OWNERS = `coord\|bot-W[1-9]…`). PR **#23** @`c24bb302` — title всё ещё с ошибочным slug. |
| **large-gate** | **Блок (steering + fam0)** | Без fam0 OOF stack A → только SMOKE. Аренда `slot:acc1/cpu` уже у bot-W4-gate, но push запрещён до fam0. |
| **sat2-gate** | **НЕ на CPU-no-net** | Есть meta+скрипт; `enable_internet=true` + `pip install ijson` при старте; датасет `elenakonkova/koz3-w2-gate` есть. Для offline CPU-акков непригодно. На боксе/акке с интернетом — зона bot-W1-sat (`box/q2` + branch lease). |
| **ae2or** | hold | Ждёт ae-dfine-gate (не трогать). |
| **ae-dfine-gate** | чужой воркер | Не трогать; 2-й GPU. |

### 1.3 Dry-run tta-gate (бокс)

Путь: `/tmp/tta-gate-dry/`  
`TTA_GATE_VARIANTS=bprime,tta_ms_sat`:

| variant | honest10 | Δ vs B′ | decision |
|---|---|---|---|
| bprime | 0.315800 | 0.0 | BASELINE |
| tta_ms_sat | 0.317900 | +0.0021 | FAIL_DELTA (порог 0.003; **dry mock**, не боевой) |

### 1.4 Bot4: `elenakonkova/koz3-w2-oof` (kaggle datasets files, токен elenakonkova)

**92 файла.** Топ-уровень: `DATASETS.md`, `lihm/`, `merged/`, `s0/`, `s1/`, `s2/`, `seedmerge/`.  
Содержимое — OOF c3-sc по сидам/фолдам + gt_li + report/region_fold.

| Искомое | Есть? |
|---|---|
| fam0 / stack A OOF (jsonl/geojson под large_objects) | **НЕТ** |
| `family_clusters.json` | **НЕТ** |

Вывод для bot4: аттач `koz3-w2-oof` **не закрывает** потребность large-gate в fam0; нужен отдельный датасет (например stack A fam0 OOF), иначе только `LARGE_GATE_SMOKE=1`.

---

## 2. Аудит оркестрации (read-only)

### 2.1 Аренды vs реальность

- **Правило «верить kernels status» подтверждено.** В HANDOFF lidar RUNNING под bot-W2 / bot-W1, но в snapshot `LEASES.json` слоты GPU часто числятся «свободен», пока не обновят set-kernel.
- На 19:21 CPU-слоты **корректно** заняты bot-W4-gate (FR1h) — это хорошо против простоя, но конкурирует с pixgbm-cr/tta.
- Просроченные branch/dataset leases (ещё в `leases[]`): `bots/w2-lidar`, `bots/w1-aeor`, `bots/w3-kurgan-seeds`, `koz3-sp-large-f0-data` (оба акка) — **expired**, мусор для check_conflicts.
- `QUEUE.json` generated **29.10** — мёртвый снимок (status=running на уже завершённом).

### 2.2 Дубли ботов / PR #16–#23 (draft)

| PR | head | Риск дубля |
|---|---|---|
| #16 aerial-recall | bot3 | закрытый рычаг ortho multiclass / ae recall — не открывать |
| #17 sp_dfine e2e-v3 | bot2 | ок как draft |
| #18 large-objects | bot4 @33791bc | пакет large-gate; **без fam0 не мерить** |
| #19 orsp_tta | bot8 | пересекается с #23 (gate поверх тех же флагов) |
| #20 pixgbm retrain | bot6 @9b411c4…37240147 | **DOC-1 риск**: retrain ≠ clean-room; tip gate package отдельно |
| #21 v4 candidate | bot5 | ок |
| #22 speedup | bot1 | ок |
| #23 tta-gate | bot8 @c24bb302 | **title/slug коллизия с ae-dfine-gate в тексте PR**; metadata локально исправлена |

Коммиты `bots/*` за 24ч: WSR STOP, orch snapshot, bot2–8 drafts (~07:33–07:47 UTC), lidar-aug e514195, плюс вечерние gate packages.

### 2.3 ERROR / CANCEL

| Kernel | Статус | Причина |
|---|---|---|
| `aaaa0000004/koz3-w2-cnx-r50c5-f01` | ERROR (~28s) | `--require-gpu: GPU нет (сессия получила CPU — второй одновременный GPU?)`. Рычаг **r50c5 закрыт** — не реран. |
| `aaaadg4gg4g4g/koz3-li-tin-59-t128-dual-t4` | CANCEL_ACKNOWLEDGED | Dual-T4 воркер на **CPU-only** акке: лог `gpus:0, gpu_names:[]`. Кто нажал Cancel в API не подписан; по смыслу — отмена бессмысленного GPU-ядра на акке без GPU (частичные `.pt` fold0–3 в output остались). На `aaaa0000004` тот же slug = COMPLETE (исторический прогон). |

### 2.4 Простои и почему

- GPU-1 обоих акков заняты lidar-aug — правильно.
- GPU-2 / TPU свободны по дизайну HANDOFF («не заполнять idle GPU без A»), пока другой воркер готовит ae-dfine-gate на 2-й GPU.
- CPU-no-net до 19:21 простаивали из-за **неготовности пакетов** (REF/CAND, fam0, internet/pip, Ae ckpt), не из-за отсутствия GO (GO на 5 акков есть).

### 2.5 Риски к 11.11

1. **v2 reproducibility на main `f127db8`:** `models/pixgbm/all/MODELS_NOT_IN_GIT.md` — артефакты PIXGBM не в git; без Drive/zip pack нельзя честно воспроизвести geo слой.
2. **S3 key rotation** — отложена до конца lidar folds (HANDOFF); keys могли утечь в source kernel versions.
3. **Clean-room vs PR#20** — путаница ролей ботов → риск «включить» EN-lineage в архив.
4. **LEASES/QUEUE drift** → ложные idle-fill и дубли push.
5. **Квота Cursor ~13%** + дубли отчётов/cloud agents → сжигание токенов без GPU-работы.

### 2.6 Устаревшие каталоги push (только список, ничего не удалял)

Старые (~29.10): `tpu-probe*`, `smoke-*`, `cnx-li-*`, `elenakonkova-*smoke`, `koz3-bot6-pixgbm-cleanroom` (старый mirror), `koz3-bot8-oof-honest10`, `koz3-dfine-ae-ft`, `koz3-dist-*`.  
Актуальные сегодня: `pixgbm-cr`, `tta-gate`, `large-gate`, `sat2-gate`, `ae-dfine-gate`, `ae2or`.

---

## 3. Топ фиксов (по убыванию impact)

1. **Синхронизировать LEASES с kernels status + чистить expired** (branch/dataset) каждый час; выкинуть/переписать `QUEUE.json` или пометить obsolete — иначе idle-fill врёт и боты дублируют работу.  
2. **Дособрать pixgbm-cr аттачи:** опубликовать/расшарить REF=`e2e-baseline-v2` zip (eb268d6) + CAND=clean-pack (`submission_pixgbm_clean_e1g1_p4_*.zip` / `models_pix_clean`) видимыми нужному акку; `enable_internet=false`; затем push на свободный CPU **после** освобождения FR1h слотов. Не мешать с PR#20.  
3. **Fam0 для large-gate:** выложить stack A fam0 OOF (+ при необходимости `family_clusters.json`) отдельным датасетом; `koz3-w2-oof` **не содержит** их — bot4 разблокируется только этим.  
4. **Расширить OWNERS в `lease.py`** (`bot-6`/`bot-8` или единый `bot-N`) либо все lease брать как `coord`/`bot-W*` — сейчас bot-8 блокируется на аренде.  
5. **Починить PR#23 текст/slug** (убрать `koz3-ae-dfine-gate` из title/body); tta реальный push только после PASS ae-dfine-gate + CPU/GPU выбор с весами без обязательного S3 на no-net акках.  
6. **Запрет GPU-ядер на CPU-only акках** в preflight (`kpool`/lease): dual-t4 на aaaadg → CANCEL с 0 GPU; r50c5 ERROR за ~30с.  
7. **S3 rotate сразу после COMPLETE lidar f0–f4**; не оставлять ключи в исходниках kernel.  
8. **Токен-экономика:** один COORD_LOG+HANDOFF владелец; не перезапускать cloud agents/отчёты-дубли; квотные watch уже hourly — не учащать.

---

## 4. PIXGBM gate numbers

Боевого `gate_report.json` / Kaggle STATUS для pixgbm-cr **нет** (пакет не пушился, REF/CAND не аттачены).  
Смежное evidence (orch sticky, не этот gate): P1 clean parity PASS; P4 zip на Drive / `out/geo-clean-pack/submission_pixgbm_clean_e1g1_p4_lleaves.zip`. Критерий будущего cr-gate: `|dF1|≤0.01` на регионах 200–217 для ALL и maggr/etgr/terfl; INVALID если пустые ступени или byte-identical при разных SHA моделей.

---

*Конец отчёта. Время бокса: 01.11.2026 ~19:22 MSK (Europe/Moscow).*
