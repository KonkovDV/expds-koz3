# HANDOFF NOW — 01.11.2026 ~18:47 MSK (coord A / bot 1 → следующий бот)
> Читать в порядке: **этот файл** → `PLAN_STICKY.md` → хвост `COORD_LOG.md` → `LEASES.json` → `bot-W*/STATUS.md`.
> Репо: https://github.com/KonkovDV/expds-koz3 · ветка оркестрации `bots/orch-protocol` · **main не пушить**.
> Секреты в текст не класть (только имена путей/env).

## 0. Жёсткие правила (не нарушать)
- Официальный LB hold **0.41238** (archive v2 eb268d6). Бот **никогда не сабмитит** на LB — только Елена загружает zip.
- FULL GO от Елены (t225/t249/t261): любые GPU/CPU/TPU/permissions; закрытые рычаги остаются закрытыми.
- Gate honest-10 vs **B′ 0.315800**: Δ≥0.003, P≥0.9, damaged не хуже. Soft-P exception: sat2 REPLACE принят при P≥0.8 (t259u «как лучше»).
- Квота Cursor ~**13%** (t264u): коротко, без fan-out; watch hourly / quota */3h.
- S3 path(b): ключи могли уехать в source Kaggle kernel versions → **rotate после lidar folds** (FULL GO t249; не блочить train).

## 1. Цель и цифры
| Метрика | Значение |
|---|---|
| Official LB (hold) | **0.41238** |
| Цель | >EN ~0.505; stretch 0.61 (PATH_061: 0.61 нереалистичен текущими рычагами, потолок ~0.42–0.44) |
| honest-10 A / B′ | 0.314615 / **0.315800** |
| Дедлайн | 11.11.2026; гейты до 08.11 |
| Локальный рост сейчас | sat2 REPLACE **0.320516** (ΔB′ **+0.00472**, P0.838 soft-PASS); lidar-only f0 **0.2892** (+0.069 vs ref), f1 **0.4969** (+0.018), mean(f0,f1)=**0.3931** early-stop PASS |

## 2. Роли ботов (Grok Bot teammates)
| Агент | id (коротко) | Роль сейчас |
|---|---|---|
| **1 / A** (этот) | coord | план, leases, сводка Елене, HANDOFF |
| **2 / B** | 09c5d647… W2 lidar | lidar R-augx на **elenakonkova** |
| **4 / D** | f92e0225… W1/W3 | sat2 + lidar f2 на **dmitrijkv**; Drive packs |
| **3** | c77110f3… | STANDBY |
| **5** | ac0b8982… | STANDBY |

Ветки: `bots/orch-protocol`, `bots/w2-lidar`, `bots/w1-aeor`, `bots/w3-kurgan-seeds`. Аренды веток через `orch/lease.sh`.

## 3. Kaggle — живой снимок 01.11 ~18:47 MSK
Доступ: `orch/pool_kaggle.sh <username> <kaggle-args…>`  
- elenakonkova ← `KAGGLE_KEY_2`  
- dmitrijkv ← `KAGGLE_KEY_1`  
(CPU-акки usertest1234 / aaaa0000004 / aaaadg4gg4g4g — через `local_ctx/ctx/kaggle_pool_4x2_t4.json`; **без интернета/S3**.)

### 3.1 Сейчас RUNNING (GPU)
| Акк | Kernel | Версия / folds | Lease | ETA / notes |
|---|---|---|---|---|
| **elenakonkova** | `elenakonkova/koz3-lidar-aug-f0` | **v5 RUNNING** c7-sc:**3,4** tip e514195 R-augx | bot-W2-lidar → **02.11 06:25** | path(b) keys in source → rotate after |
| **dmitrijkv** | `dmitrijkv/koz3-w1-lidar-aug-f2` | **v1 RUNNING** c7-sc:**2** tip e514195 scratch | bot-W1-aeor → **02.11 06:28** | ETA ~**21:45**; pull → `out/li_aug/f2_dkv/` |

Проверка:  
`orch/pool_kaggle.sh elenakonkova kernels status -k elenakonkova/koz3-lidar-aug-f0` → RUNNING  
`orch/pool_kaggle.sh dmitrijkv kernels status -k dmitrijkv/koz3-w1-lidar-aug-f2` → RUNNING

### 3.2 Недавние COMPLETE (не перезапускать)
| Kernel | Статус | Итог |
|---|---|---|
| elenakonkova/koz3-lidar-aug-f0 **v2** | COMPLETE | f0 Li-only F1 **0.2892** vs ref 0.2198; out `out/li_aug/v2` |
| elenakonkova/koz3-lidar-aug-f0 **v4** | COMPLETE | f1 Li-only **0.4969**; mean 0.3931 **early-stop PASS**; out `out/li_aug/v4` |
| dmitrijkv/koz3-w1-sat2-f14 | COMPLETE | folds1–4 no-TTA geojsons 175/1086/206/286; out `out/sat2-f14/kout` |
| dmitrijkv/koz3-w1-sat2-f0 | COMPLETE | fold0 damaged F1 0.4175; merge-gate FAIL; replace-path → soft-PASS |
| dmitrijkv/koz3-tpu-smoke | COMPLETE/dead | cancelled stub |
| CLOSED STOP | r50c5, WSR f127db8, ring/near_kept, peak-rerank, dino-gor, sp_large | **не реран** |

### 3.3 Свободные слоты (не заполнять без A)
acc1/gpu-2, acc1/tpu, acc2/gpu-2, acc2/tpu, CPU-акки — свободны. LEASES.json на боксе может отставать от живых kernel status — **верить `kernels status`**.

### 3.4 Как пушить / арендовать
1. `orch/lease.sh take slot:… → bot-… until …`  
2. Push только под арендой; slug для dmitrijkv W1: **`koz3-w1-*`** (lease prefix).  
3. После COMPLETE: pull output → `out/…`, release lease, STATUS.md + строка в COORD_LOG.

## 4. Артефакты для LB (Елена uploads; бот не сабмитит)
Drive folder: https://drive.google.com/drive/folders/17N5Cu1bZJ38iYu3W_nCN3zOk6R3l5riX

| Pack | Где | sha / note |
|---|---|---|
| **sat2 REPLACE soft-PASS** | Drive + `out/sat2-f14/replace-pack/sat2_det2_oof_replace_pack.zip` (~1.1MB) | sha `c1e54f76…`; score 0.320516 |
| **PIXGBM clean P4** | Drive 7×parts + `out/geo-clean-pack/submission_pixgbm_clean_e1g1_p4_lleaves.zip` | sha `ed51cae4…`; filters OFF; budget 900 |
| **FR1h** | Drive 6×90MB parts | sha `fe8d60b9…`; zip `out/fr1h-pack/submission_fr1h_on_eb268d6.zip` |
| **FR1h+M_rp** | Drive 6×90MB parts | sha `e5b6c3b8…`; zip `out/fr1h-mrp-pack/…` |
| verify-spor | `out/verify-spor-pack/` | готов, LB-probe |

Кандидаты LB по приоритету аудита: FR1h → FR1h+M_rp → PIXGBM (DOC-1 её call) → sat2 replace (после сборки в submit stack).

## 5. Что делать сразу после входа
1. Прочитать этот HANDOFF + `PLAN_STICKY.md`.  
2. `pool_kaggle` status на оба RUNNING; если COMPLETE → pull + early-stop/mean на f2/f3/f4.  
3. Когда **все** lidar folds f0–f4 готовы → 5-fold Li metrics + (опц.) honest-10 lidar arm; **S3 key rotate**.  
4. Не заполнять idle GPU без явного GO.  
5. Не сабмитить LB.  
6. Закрытое не открывать: r50c5, WSR, ring, peak-rerank, dino-gor folds, sp_large, sat 8m cluster.

## 6. Бокс-пути
- Рабочая зона: `/workspace/koz3/` (orch, out, expds-koz3 wt, repo clone).  
- Lidar outs: `out/li_aug/{v2,v4,f2_dkv,…}`  
- Sat2: `out/sat2-f0/`, `out/sat2-f14/`  
- Plans: `orch/BEAT_EN_*.md`, `orch/plans/`, `orch/PATH_TO_061.md`  
- Auth Kaggle: env KEY_1/KEY_2 + `local_ctx/ctx/`  
- S3 keys file (не в git): `/workspace/koz3/s3_keys.tsv`

## 7. Routines (Grok Bot A)
- `koz-3-beat-en-5m-watch` — **hourly** (было 15m; квота).  
- `koz-3-quota-idle-fill` — каждые **3h**.  
- Писать Елене только COMPLETE/STOP/блокер.

## 8. Следующий рост по баллам (приоритет)
1. Дожать lidar f2+f3+f4 → полный R-augx OOF / mean.  
2. Елена upload FR1h / sat2 / PIXGBM (локальный + уже на Drive).  
3. После lidar: site-balance / 3-seed blend (BEAT_EN G2/G3) — только с lease.  
4. S3 rotate path(b).

---
Снято coord A 01.11.2026 ~18:47 Europe/Moscow. Обновлять этот файл при каждом COMPLETE/STOP.
