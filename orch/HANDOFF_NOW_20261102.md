# HANDOFF NOW — 02.11.2026 ~18:40 MSK (deep freeze, coord A → любой следующий бот / другой аккаунт)

> **Читать первым.** Порядок: этот файл → `STATE.md` (NOW-шапка) → `FINAL_CANDIDATES.md` §0–§1 → `HANDOFF_NOW_v4.md` → `PLAN_STICKY.md` → `ROSTER.md` → `PROTOCOL.md` → `bot-W4/STATUS.md` (хвост) → хвост `COORD_LOG.md`.
> Репо: https://github.com/KonkovDV/expds-koz3 · ветка оркестрации **`bots/orch-protocol`** (этот файл лежит в `orch/`). **main не пушить, force-push никогда.**
> Секреты в текст/git/Drive не класть — только имена env/путей.

## 0. Жёсткие правила
1. **LB сабмитит только Елена.** Бот никогда не загружает zip на платформу. Официальный LB hold **0.41238** (v2 eb268d6, 29.10).
2. **Git:** работать в worktree `/workspace/koz3/wt/orch` (ветка `bots/orch-protocol`); `main` не трогать; **никогда `--force`**; перед каждым коммитом secret-scan: `grep -rE 'KGAT_[A-Za-z0-9]{8,}|ghp_|github_pat_|AKIA[0-9A-Z]{12}'` по добавляемым файлам.
3. **GPU** только на аккаунтах **elenakonkova** и **dmitrijkv** и только по аренде от coord (`orch/lease.sh`, `LEASES.json`). CPU-пул: usertest1234 / aaaa0000004 / aaaadg4gg4g4g (без интернета/S3).
4. **Гейты — только независимые, W4** (бот-гейт, harness `orch/bot-W4/comp/*`, honest-10). Автор рычага сам себе гейт не засчитывает.
5. **Бар гейта (всё одновременно):** Δall ≥ **+0.003**, P ≥ **0.95** (при семействе — Бонферрони, напр. 0.975 для m=4), **new34 ≥ 0**, **nolidar ≥ 0**, floors ΔF1: kp/kt/gor ≥ **−0.002**, arkh/fort ≥ **−0.005**. Провал любого пункта = OFF.
6. **Post-hoc варианты** (найденные в отчёте гейта) — только через **новый PREREG** до замера, иначе это подгон.
7. Квота Cursor ~**20%**: без fan-out, без новых GPU-кернелов без вектора Елены.
8. Секреты: Kaggle токены — `~/.secrets/kpool/<user>` → env `KAGGLE_API_TOKEN` (через `orch/pool_kaggle.sh <user> …`); S3 — `local_ctx/s3_train.env`. **Не копировать** в git/Drive/доки.

## 1. Цель и цифры
| Метрика | Значение |
|---|---|
| Official LB (hold) | **0.41238** (v2 eb268d6); v3/v4 на LB **ещё не отправлялись** |
| honest-10 база v3_on (b447649, прод-реплей) | **0.345439** |
| honest-10 **v4 (C1F-all)** | **0.361641**, **Δ+0.0162 P1.0** vs v3_on, new34 +0.0007 |
| Старые базы | A 0.314615 · B′ 0.315800 · v2clean 0.313967 |
| Дедлайн | 11.11.2026 (freeze 10.11 23:59 MSK) |
| Цель | >0.505 (EN), stretch 0.61 — `PATH_TO_061.md`: текущими рычагами нереалистично |

## 2. Финальные артефакты (PASS)
| Роль | Архив | sha256 | Где |
|---|---|---|---|
| **v4 PRIMARY** = v3_on b447649 + scene_drel 98cdf56 + **C1F-all** (`configs/v4_c1f.toml` @ `bots/w4-v3on` **07795d2**) | `koz3_v4c1f_07795d2_51882efd_w6dfd2e07.zip` (722 705 579 B, 3904 файла) | `f0def9822cf185aee0829b72c36e4def94c17d03044d16adefc59c9547202cef` | box `/workspace/koz3/w4geo/dist/` (+ `v4_drive_parts/`); Drive [v4c1f_f0def982](https://drive.google.com/drive/folders/1h8AAWDV6xJkgxN7FwZXeLMYaAIEUx5Ia) (8×part + SHA256SUMS); Kaggle DS `aaaa0000004/koz3-w4-v4c1f` |
| rollback 1: v3_on b447649 | `koz3_v3on_b447649_6d4764f8_w6dfd2e07.zip` | `1406c01a9aaafc996862a14f12cde14f6bf05b313ad0bbce14428df1d10ef41d` | box `v4build/in/…zip.bin`; Drive [v3_on_b447649](https://drive.google.com/drive/folders/1PcLcJ7JacT5CDepcVq71v61lbqlK4SBC) |
| rollback 2: v2clean | `koz3_v2clean_70d335a_c028f4eb_wb042594f.zip` | `545ef32a70b4994f4d694828ae5eb28f27c9161d7f19a96f5a9d3833b87e093f` | box `w4geo/kaggle/ds_v2clean/`; Drive [v2clean_545ef32a](https://drive.google.com/drive/folders/1omTT3QFxb3zgFAFQjj862Q6LkQ92uBhQ) |

- **v4chk PASS** (Kaggle `aaaa0000004/koz3-w4-v4chk`, 02.11 ~15:35 MSK): D/C/A/B среды байт-идентичны (smoke `2fea6d16…`, ref103 `2bd9a361…`, n_det 155), нет V3_DEGRADED; REF_v3on отличается ровно C1F; NEG_dfine_broken FAIL-as-designed. Outs `/workspace/koz3/out/v4chk/`.
- Гейт v4: `v4build/gate/` (c1fx_gate.json = 607edef), out_C1 sha `3d286d2212ac73fdbeb839aa30e612af054077a3eb19f812008b404ebc0c2ecc`.
- Drive-корень архивов: KOZ3-final https://drive.google.com/drive/folders/1GzW2XGWIMVybFV2pP3WLryeIIhQeN8NB (приватно); reports https://drive.google.com/drive/folders/1-cMktRigtZuZRN7peDVqzJe3YNxZFtnP.
- **Порядок сабмита (Елена):** v4 → если LB хуже → v3_on (1406c01a…) → v2clean (545ef32a…).
- Склейка: `cat <name>.zip.part0* > <name>.zip && sha256sum -c SHA256SUMS`.

## 3. Закрыто 02.11 (все FAIL vs v4 / на v4-харнессе) — пути в `orch/bot-W4/`
| Рычаг | Итог | Ключ | Файлы |
|---|---|---|---|
| GORLOC L1/L2/L2×m | OFF | Δ+0.0013/+0.0018/+0.0039, P0.60–0.81; L2 new34−; Kislovodsk LOFO −0.0024 | `runs/regate_COMP_GORLOC_KMASK_CLOSE.{json,md}` |
| KMASK | NO-GATE | oracle ceiling Δ+0.00065 < 0.002 | там же |
| COMP_FL (sat_frame t2_cls) | OFF/CLOSED | Δ+0.00012 P0.53; nolidar −0.0028; kp floor −0.0027 | `runs/regate_COMP_FL.json`, `runs/regate_COMP_FL_notes.md`, `comp/fl_*.py` |
| C1F+FP1 → v5 | OFF (v5 не формируется) | vs v4 Δ−0.00276 P0.037; kt floor −0.00227 vs base | `runs/gate_C1FP_v5.{json,md}`, `comp/c1fp/` |
| satdet-frcnn (W1 PREREG 843d807) | OFF | vs v3_on P1/P2 Δ+0.0134 P0.996 но new34 −0.0027; D1/D2 floors/P; vs v4 C1D1f Δ−0.0019, C1D2f Δ+0.0014 P0.58 gor −0.026 (arkh +0.05) | `runs/gate_satdet_frcnn.{json,md}`, `comp/satdet/{out,outc1}/` |
| dfine2-a (18/132 рег.) | OFF | vs v4 Δ+0.00041 P0.56; arkh −0.0078; early_stop PASS | `runs/gate_dfine2a_vs_v4.{json,md}`, `comp/dfine2a/` |
| dfine2 full ENS8 (132/132) | OFF | vs v4 **Δ+0.00525 P0.9545** но new34 −0.00116, kt −0.0030, arkh −0.0221 | `runs/gate_dfine2_full_vs_v4.{json,md}`, `comp/dfine2full/` |

Паттерн: против старого v3_on многие рычаги дают +0.01…+0.02, но **поверх v4** (C1F-all уже съел центр-приор) либо ≈0, либо ломают new34/kt/arkh/gor.

Ранее закрыто (01.11 и раньше, **не реранить**): J2/RR1s-cf, CTXc, lidar-aug, r50c5, G1 sat2-kp, G2 li nms8, large-gate, ae-dfine-gate, v2+sat2, M_rp поверх V3P, V7 D-FINE, c3-sc, WS-RR1/2, xlayer, WS-G1 ViT-S, peak-rerank, WSR, ring, dino-gor, sp_large, Sr8G/noTerFl (C*). Детали: `FINAL_CANDIDATES.md` §2, `STATE.md` ARCHIVE.

## 4. Открытые ожидания (нужно «да»/вектор Елены)
1. **LB submit v4** — только Елена (zip из Drive v4c1f_f0def982 или box).
2. **S3 rotate** ключей (могли попасть в source Kaggle kernel versions) — ждёт «да».
3. **Disk UNSURE** (ничего не удалять без «да»): `w4geo/work` 22G (clean-room pixgbm, E1_G1 frozen), `w5data/sat_raw` 5.2G, `v4build/v3on_x` 762M, `orch/bot-W1/e2e_v7` ~3.6G, `out/{li_aug,r50c5-*,sat2-f14,fr1h-gate}`, `w4geo/kaggle/ds-v3a-min-check`. Отчёт: `/workspace/koz3/out/audit-next/DISK_AUDIT_20261102.md` (+ `DELETE_LOG_20261102.txt`, `sha_proof_20261102.txt`). Диск `/` сейчас ~73% (36G free).
4. **Новый PREREG** — кандидаты post-hoc (только если Елена даст вектор): arkh-only sp_det2 (satdet FRCNN помогает arkh +0.05…+0.09), ENS8 без arkh в dfine2-merge (gor +0.043, nolidar +0.0098).
5. Routines beat-EN / quota watch — на паузе; W4 STANDBY; GPU-рычаги исчерпаны.

## 5. Роли (ROSTER, актуально 02.11)
| Агент | Роль | Сейчас |
|---|---|---|
| A (coord, бот 1) | оркестрация, STATE/QUEUE/LEASES, сводка Елене, HANDOFF | STANDBY, этот freeze |
| B (агент 2, 09c5d647…) | W2 лидар (dfine2-a/b PREREG P-W2-13) | done/STANDBY |
| C (агент 3, c77110f3…) | старый W4/W5 гейт+реранкер | STANDBY |
| D (агент 4, f92e0225…) | W1/W3 аэро/орто, курганы, satdet-frcnn | done/STANDBY |
| W4 (агент 5, ac0b8982…) | независимые гейты, сборки v4, parity | STANDBY |
Новому боту: взять роль A (coord) + при необходимости W4-гейт. Аренды веток — `orch/lease.sh`.

## 6. Синхронизация book ↔ git
- Живая книга: `/workspace/koz3/orch` (НЕ git). Git-копия: `/workspace/koz3/wt/orch/orch/`.
- Синк (без шума и тяжёлых данных):
  `rsync -a --max-size=5M --exclude='.queue.lock' --exclude='.watcher.pid' --exclude='.leases.lock' --exclude='*.bak*' --exclude='tmp_*' --exclude='*.geojson' --exclude='*.pkl' --exclude='work/' --exclude='x_v3on/' --exclude='pydeps/' --exclude='e2e_v7/' /workspace/koz3/orch/ /workspace/koz3/wt/orch/orch/`
  затем в `wt/orch`: secret-scan → `git add orch` → commit → `git push origin bots/orch-protocol` (без force). `.gitignore` репо уже режет *.zip/*.pkl/*.npy/*.tif/данные/секреты.
- На новом аккаунте/боксе: `git clone -b bots/orch-protocol https://github.com/KonkovDV/expds-koz3.git` → `orch/` — то же содержимое (без данных; данные — S3/Kaggle DS/Drive).

## 7. Варианты следующего вектора (решает Елена)
a) **Сабмит v4** на LB (рекомендовано: единственный PASS-кандидат, +0.0162 honest-10, parity PASS).
b) Новый PREREG на arkh-only/ENS8-без-arkh (CPU, на готовом harness `comp/dfine2full`, `comp/satdet`) — дёшево, ~1 гейт.
c) Снос UNSURE диска / S3 rotate — гигиена.
d) Снять STANDBY / beat-EN — только при росте квоты.

## 8. Ключевые пути box
`/workspace/koz3/w4geo/dist/` (v4 zip) · `/workspace/koz3/v4build/{gate,res,in}` (живые входы гейта) · `/workspace/koz3/orch/bot-W4/{runs,comp}` · `/workspace/koz3/out/v4chk/` · `/workspace/koz3/out/audit-next/` · `/workspace/koz3/out/handoff-20261102/` (этот пакет; tar.gz + HANDOFF + TRANSFER + MANIFEST на Drive https://drive.google.com/drive/folders/1IzWEzCFzJ86UKX9KYSDS1fFuOwLQFj3N) · `/workspace/koz3/repo` (main read-only).
