# PATH_TO_061 — KOZ-3: путь к 0.61 + статус всех потоков (30.10.2026 13:10 MSK, A/coord-exec)
Источники: OFFICIAL_SCORES, STATE, MASTER_PLAN, REVIEW_SOTA, WEAKSPOTS_PLAN, AUDIT_SEAMS, PROTOCOL, SCHEDULE, HANDOFF, QUEUE.json, LEASES (lease.sh list), bot-W*/STATUS, PREP_*, wt/w2/HANDOFF_W2.md + PREREG_W2_cnx.md, Kaggle status (read-only 13:03). Новых баллов не придумано; производные арифметические оценки помечены «расчёт».

## 0. Главное за этот проход
- **Блокер table.pkl/OOF A СНЯТ без ПК.** Полная база A лежала в приватном датасете `elenakonkova/koz3-w2-gate` (`w5base/`). Скачано на бокс:
  - `/workspace/koz3/w5data/stack_arch.geojson` (1.13 ГБ), `/workspace/koz3/w5data/gt_train_v2.geojson`, `/workspace/koz3/w5data/work/table.pkl` (364 МБ), `/workspace/koz3/w5data/work/base_counts.pkl`.
  - Проверка: `gate_v6c.py --vs base_counts.pkl` на боксе → honest-10 **0.3146153226828423**, 103 семьи, d=0 (= A). Лог `w5data/base_repro_A.log`.
  - venv бокса был пуст (бокс пересоздан) → доставлены numpy/scipy/shapely/pyproj/lightgbm/pandas.
  - Там же готовые W2-гейты (s2, s012, varb, varb_kur) и `elenakonkova/koz3-w2-oof` (OOF c3-sc).
- Осталось недоступным: `d3_gate.py` (нигде нет). check4 PASS (30.10 13:28). Пакет Елене: `w4geo/dist/koz3_v3a_min_4da2eb9_*.zip` (v2clean+ORG3+M_rp).
- Закрыто с момента MASTER_PLAN (29.10): li_hm heatmap (P-W2-7, ранний стоп FAIL: Li-only f0/f1 0.3137 ≤ 0.3492), center snap курганов (P-W2-4 FAIL, нетто −6), DEM fill (P-W2-10 FAIL, Δ+0.0014), c3-sc (P-W2-2 Δ−0.0162; s012 Δ−0.0071; (b) Δ−0.0195). Рычаг «Lidar heatmap +0.02–0.04» из моста MASTER_PLAN выбывает.

## 1. Разрыв
| Что | Значение | Источник |
|---|---|---|
| Лучший LB | 0.4124 (29.10, v2 eb268d6) | OFFICIAL_SCORES |
| Цель | 0.61 → нужно +0.1976 | расчёт |
| Офлайн A (honest-10) | 0.314615 | STATE, подтверждено на боксе 13:08 |
| Потолок идеальных порогов | 0.367 | REVIEW_SOTA §2.2 |
| Потолок оракула-реранкера | 0.658 | REVIEW_SOTA §2.2 |
| Подтверждено к A (PASS) | ≈ +0.028 | STATE |
| Прогноз MASTER_PLAN | ≈0.47–0.48; P(>0.5)≈25%; P(≥0.61)<3% | MASTER_PLAN §2 |
| Тот же мост без heatmap (+0.02) | ≈0.46 | расчёт: 0.493 − 0.02 − 0.015 |
Офлайн→LB: v2 gate +0.011/+0.0139 → LB +0.0644; arch +0.0036 → LB +0.0144; 24.10 self-fit 0.3538 → LB 0.3285 (cross-family 0.3295 совпал). Шум LB ±0.034; закрытый тест ±0.05 и другой состав (публичный: геофизика ~50%, лидар 33%, аэро 2.6%, орто 1.7%). Gate недооценивает модальностные сдвиги → нужен gate по составу (#12).
Вывод: 0.61 на публичном LB не достигается текущими рычагами. Реальный резерв: (а) ранжирование (оракул 0.658 vs 0.367), (б) закрытый тест (ae/or, без лидара: FN 0.38).

## 2. Подтверждено (PASS)
| Что | Δ | Источник |
|---|---|---|
| ORG3 EtGr 212–217, TerFl off (nested-fold) | +0.0232 (регейт coord: +0.0267, P_family 0.97) | STATE; bot-W4/runs/independent_regate_ORG3.json |
| M_rp: micro_circles→rp (gor/arch/obj/fort) | +0.0044 (P 0.989; new34 +0.0022) | AUDIT_SEAMS #1, bots/w5-rerank c4e18a9 |
| Архив v2clean 545ef32a (DOC-1 guard) | check4 PASS FAIL0; пакет v3a_min (+M_rp) sha132c22eb готов | STATUS W4 |
| Постпроцесс shapely 28→6 с, Rust e2e ×2.5 | инфра | SCHEDULE, STATE |

## 3. Отклонено — не повторять без новых данных
V7 D-FINE ae+or (Δ+0.0033, P 0.935 < Бонферрони 0.9857); c3-sc аддитивно/замена; li_hm; center snap курганов; DEM fill; seed-ансамбль c3-sc; chain2 V5 (−0.0271)/V5x (−0.0112)/V6c (−0.0103); WS-RR1/RR2; WS-A1; K1–K5/KT; H1–H5; V8 max_det 300; «кроны в DEM»; негатив-лист REVIEW_SOTA §2.4 (MORPH, ADAF, WBF, coarse_hillfort −0.016, gormask −0.0056, kurgan ae c7g −0.014, match_scope=layer −0.0206 …).
Под вопросом: D-FINE SpOr (gate +0.0198 vs mock-e2e −0.0023); gor_dense = NO-OP (не включался).

## 4. Открытые ставки (Δ — оценки из документов, не замеры)
| # | Ставка | Δ ожид. | Цена | Риск | Зависимость | Владелец |
|---|---|---|---|---|---|---|
| 1 | W4 паритет v2 (golden e2e) | защищает 0.012–0.06 | CPU 4–6 ч | низк | table.pkl ✓ (снят) | C |
| 2 | v3a CPU-рычаги на OOF A: break-even τ (+0.005–0.02), xlayer Li→Ae/Sp (+0.005–0.02), object_poly/MagGr 0.05 (+0.01–0.02; G2 +0.0099), multihyp (+0.005–0.015), resweep после M_rp (≥0), dedup (+0.001) | сумма +0.02 (MASTER_PLAN v3a 0.432) | CPU | ср | OOF A ✓, PREREG | C |
| 3 | Gate по составу теста (#12) | инструмент | <1 CPU-ч | низк | OOF A ✓ | C |
| 4 | WS-G1 kNN на dino-emb для gor/fort + реранкер с гейтом по модальности | +0.002…+0.008 / +0.01–0.04 | CPU 1 ч + LGBM | ср | dino-emb ✓ на боксе | C |
| 5 | Ae all-class D-FINE (закрытый тест) | +0.005 публ / +0.02–0.05 закр. | 2–4 GPU-ч | ср | e2e V7 sign check (D), d3_gate | D |
| 6 | r50c5 P-W2-9 (2-й лидар-детектор) | не оценено; ранний стоп 0.3492 / 0.2369 | f01 2.6–3.3 GPU-ч + f234 ≤11.3 | ср | после 03:05 | B |
| 7 | CI2-1 cloudfeat как признак | +0.003…+0.01 | CPU | ср | cloudfeat COMPLETE | C/D |
| 8 | CI2-5 adacrop gor/fort | +0.002…+0.008 | GPU 2–3 ч | высок | код, кэш тайлов, PREREG | D |
| 9 | CI2-7 сид×постпроцесс; kurgan seeds OOF; conf-ties (#7) | 0…+0.005 | CPU | низк | seeds COMPLETE | D/C |
| 10 | Sat recall (#11) | +0.003–0.01 | 1–2 GPU-ч | ср | после 03:05 | D |

## 5. Блокеры
| Блокер | Статус | Кто снимает |
|---|---|---|
| table.pkl/OOF A/stack_arch/GT | **СНЯТ 13:08** (koz3-w2-gate) | — |
| d3_gate.py | нет ни на диске, ни в git | D пересобирает по ae_final_prereg_wsa1.json, или ПК |
| Ключи CPU-пула (kaggle_pool_4x2_t4.json) | нет на боксе; ядра coord на usertest/aaaa/aaaadg запущены до пересоздания (статус не проверить) | Елена |
| check4 output (aaaa0000004) | PASS 30.10; v3a_min zip готов для Елены | C / Елена |
| GPU-квота | acc2 GPU 0 до Сб 03:00; acc1 ≈3.1 ч на 10:32 (COORD_LOG), минус dino-emb-gpu RUNNING | сброс 31.10 03:00 MSK |
| Датасеты *-s3-cred на elenakonkova и dmitrijkv | существуют (п.13.10, риск) | Елена решает |

## 6. Роли и следующее действие
| Агент | Потоки | Следующее |
|---|---|---|
| A coord | STATE/QUEUE/leases | (1) снять hold `w0s-resweep-micro-rp`, `w0s-conf-ties` → ready (OOF A есть); (2) решить D0: отмена 2 TPU-смоуков QUEUED без аренды; (3) решить отмену dino-emb-gpu (CPU-версия COMPLETE, жжёт acc1 GPU); (4) освободить stale box/cpu W2 (P-W2-10 завершён); (5) продлить branch:bots/w2-lidar для B под P-W2-9 |
| B | W2 | ждать 03:05; P-W2-9: lease → push f01 (acc1/gpu-1) + f234 (acc2/gpu-1) → ранний стоп → FAIL ⇒ cancel f234 |
| C | W4+W5 | сейчас на box CPU: паритет v2 (S0-box1), gate по составу #12, v3a рычаги, WS-G1 на dino-emb; check4 — после ключа |
| D | W1+W3 | e2e V7 sign check на OOF/весах (уже скачаны) + table.pkl ✓; восстановить d3_gate; CI2-5 код+PREREG; seeds → CI2-7 |

## 7. Загрузка (цель: ни одного простоя)
| Ресурс | Сейчас | Решение |
|---|---|---|
| acc1 gpu-1 | свободен (cooldown 03:05) | **беречь под r50c5 f01** |
| acc1 gpu-2 | dino-emb-gpu RUNNING (дубль) | coord: отменить → сберечь квоту |
| acc1 tpu | koz3-acc2-tpu-smoke QUEUED, аренда истекла 12:16 | D0: отменить; TPU-готового кода нет → простой честный |
| acc1 cpu | kurgan-cpu-seeds RUNNING | ок |
| acc2 gpu-1/2 | квота 0 | HOLD до 03:05 (r50c5 f234, затем gordense-timing) |
| acc2 tpu | koz3-tpu-cnx-c3sc-train QUEUED (c5 закрыт) | D0: отменить |
| acc2 cpu | kurgan seeds 69,70 до 17:14 | ок |
| usertest/aaaa/aaaadg cpu | аренды coord (seeds), статус не виден | ключи от Елены |
| box (8 ядер, ≤3 тяжёлых/бот) | load 0.4–0.9 = простой | C: паритет + v3a + #12; D: e2e V7; A: resweep micro-rp |

## 8. Расписание
- До 31.10 03:05: только CPU/box/подготовка; никакого GPU-push.
- 03:00 сброс → 03:05 рутина: B P-W2-9 (f01 acc1/gpu-1, f234 acc2/gpu-1) → ~05:40–06:30 ранний стоп f01 → FAIL: cancel f234, квота W1/W3; PASS: f234 до ~14:30 → гейт C.
- acc2/gpu-2 после 03:05: gordense-timing (dmitrijkv, аренда A) или CI2-5 (elenakonkova).
- LB: никаких сдач без PASS паритета v2 (W4) и решения Елены. 3 сдачи/день максимум; жмёт только Елена.

## 9. Ответ D (для SendToAgent)
см. отчёт A: (1) ДА pull ae-final/or-final (CPU, lease bot-W1-aeor); (2) table.pkl/GT/stack_arch теперь на боксе (`w5data/...`), d3_gate.py нет — пересобрать; (3) CI2-5 остаётся на elenakonkova (S3/интернет elena или шаринг xfer на elena), не переносить молча на dmitrijkv; gordense-timing — dmitrijkv, hold до 03:05 + аренда A.

## C progress (30.10, 13:30 MSK, bot C) — all numbers measured on OOF A (honest-10, 103 families, boot 2000), none estimated
**Parity v2: PARTIAL (pass-with-caveat).** v2 release artifacts in git eb268d6 match REPRO_V2 sha manifest; golden e2e cannot be re-run on box (no LAS/tif inputs, no v2 zip). Prior byte-identical result (29.10, 6 sites) stands. LB blocker: v2clean 545ef32a check4 (aaaa key). `bot-W4/runs/parity_v2_20261030.json`
**#12 composition gate:** implemented (public mix geo .50/li .33/sp_or .127/ae .026/or .017; closed stress mix, assumption). A public-weighted = 0.35998. `bot-W4/runs/compgate_notes.md`
| lever | Δ | P | new34 | optical | #12 pub (P) | #12 closed (P) | verdict |
|---|---|---|---|---|---|---|---|
| M_rp (v3a-min, reproduced) | +0.00437 | 0.989 | +0.0022 | +0.0085 | +0.0021 (0.92) | +0.0038 (1.00) | **PASS** |
| X1 xlayer Li→Ae/Sp, Sp↔Or | −0.0110 | 0.01 | −0.018 | 0.000 | −0.017 | −0.007 | FAIL |
| L3be break-even τ (isotonic) | +0.0015 | 0.87 | +0.0031 | 0.000 | +0.0020 (0.85) | +0.0010 (0.85) | FAIL |
| L3m τ per site-modality | +0.0080 | 0.80 | −0.054 | +0.014 | +0.0074 | — | FAIL |
| L3a τ per area bin (large obj) | −0.0035 | 0.01 | −0.010 | 0.000 | −0.005 | — | FAIL |
| L10o object_poly >30 m | +0.00001 | 0.63 | 0 | 0 | 0 | — | FAIL |
| L14 / L14s / L14g dedup | −0.019 / −0.009 / 0 | ≤0.02 | | | | | FAIL |
| WS-G1 / G1n (ViT-S kNN gor/fort) | −0.0216 / −0.0109 | 0.00 / 0.04 | −0.05 / −0.03 | | | | FAIL |
Stack v3a = M_rp only (sole PASS) → `bot-W4/runs/v3a_stack_gate.json`. ORG3 is on a separate 18-site geo OOF, not in this table.
**Why xlayer cannot work:** official compute_metrics matches globally per class (markup_type only for reference stats) — a Li pred already hits Ae GT at the same spot; a copy only adds FP. Measured twice (V3 −0.0098, X1 −0.0110).
**Aerial / large objects (oracle = any candidate hits):** ae povr 0.45, ae tselye 0.45, ae gor 0.37, ae arch 0.00, ae fort 0.65 → 639/1162 ae povr GT have no candidate at all ⇒ aerial is detector-bound (W1 D-FINE ae, GPU), no post-proc lever. Large GT: gor ≥5k m² 0.44–0.45 oracle, fort ≥5k 0.82–0.91, povr ≥5k 0.63–0.78.
Blockers: MagGr 0.05 m / L8 EtGr depth / L13 contour need raw geophysics inputs (not on box). check4 PASS; LB#1 package v3a_min ready for Elena.
