# Бот 6 — clean-room PIXGBM (отчёт 29.10.2026, Europe/Moscow)

## Контракт
| Поле | Значение |
|---|---|
| **Цель** | Clean-room переобучение PIXGBM за флагом OFF; ядро Kaggle на TPU-VM как CPU-хост с сборкой дерева из xfer-шардов. |
| **Гипотеза** | С deterministic-параметрами синтетика бит-в-бит; на полном КОЗ-3 sha256 эталона маловероятен → metric-fallback. |
| **Критерий приёмки** | **(b) e2e-metric** (после FAIL sha @b0add08): mock + honest-10 с новыми весами vs эталонные `models/pixgbm/all`; Δ≥−0.01 и послойно no-drop >0.01 (PROTOCOL). Byte/sha match **снят** как недостижимый. |
| **Бюджет** | ПК CPU 12 потоков (оценка 15–60 мин wall); Kaggle TPU-VM отменён (данные). |
| **Ветка** | `bots/bot6-pixgbm-cleanroom` @ `b0add08` (3f61ea8 + pick_rasters 1-arg) → draft PR https://github.com/KonkovDV/expds-koz3/pull/5 |

ПК — основной путь: `pc/QUEUE.md` [bot-6].


---

## Диагностика FAIL @b0add08 (29.10 ~18:24–18:35 МСК) — путь **(b)**

### Факт прогона (изм.)
| Поле | Значение |
|---|---|
| Job | `C:\Sloi\pc_jobs\bot6_pixgbm_b0add08` |
| Verdict | FAIL: sha mismatch; max\|Δp\| ≈ **0.446** (poly) / **0.566** (finds) vs tol 1e-4 |
| Совпало | feature_names (72), 300 деревьев, `--match-reference`, geo sha `35e4331b…`, lightgbm 4.7.0, threads=2 |
| Новые sha | poly `c0c1c7de…`, finds `c338126b…` (≠ эталон `d1fe3efa…` / `5fa8c09d…`) |
| Bot-1 pytest | 674 passed, 0 failed; EXPERIMENTS.csv строка 18:24 уже есть — **не дублировать** |

### n / pos (изм.)
| Модель | REF `train.log` (bot-3) | NEW `train_info.json` @b0add08 | Δ |
|---|---|---|---|
| all/poly | n=**949004**, pos=**353896** | n=**948974**, pos=**354241** | Δn=−30, Δpos=+345 |
| all/finds | n=**201716**, pos=**38640** | n=**201606**, pos=**38624** | Δn=−110, Δpos=−16 |

Тот же n/pos у @d173a75 (повтор cleanroom) → extract+train детерминированы на текущих данных, но **≠ эталону**.

### Extract: растры и per-column (изм.)
- NEW extract: **92** растра, 18 регионов, sum_n=1002118, poly_pos=354241, finds_pos=38624; mt: etgr=76, maggr=14, terfl=2 (incl. **207**).
- REF `extract.log`: **91** JSON-строк + skip 207 «нет разметки»; sum_n=985997; после `extract207.log` 207: n=15924, poly_pos=**5473**, finds_pos=**3127**, n_gt_poly=40, n_gt_finds=427.
- NEW 207: n=15911, poly_pos=**5576**, finds_pos=**3092**.
- Per-raster drift примеры (REF → NEW): 209 n 15815→15796 pp 6074→5967; 208 pp 4198→4212; 211 серый n 15630→15599.
- `feature_infos` REF vs @d173a75 (тот же n/pos что b0add08): **67/72** признаков с разными min:max (крупнейшие |Δ|: `zd_log0.4`≈3.23, `zd_hp1.5`≈1.96, `r2.0_opp`≈1.95). Tree=0 split_feature/threshold расходятся после первых узлов.
- Код признаков **байт-идентичен** `441bb71`: `load_map` / `features_at` / `features_for` / `label_masks` / `pick_rasters` / `sample` / `cmd_extract` / `fit` (AST-sha совпали с vendored geo `35e4331b` + scripts @82d23a0 для extract/fit).
- `80d50b88` hp_energy: default `gray` ≡ 1-arg; эталон обучен **до** него (441bb71/82d23a0 28.10 вечер). **Не причина Δp.**
- Эталонные npz feats в архиве bot-3 **отсутствуют** (только `extract_log.json` на 1 строку + extract.log). Снимка данных REF нет.

### Ранжированные гипотезы + дешёвые проверки
1. **Дрейф матрицы признаков (разметка / valid∖noise / пулы sample) — ОСНОВНАЯ.** Док-во: per-raster n/pos ≠; feature_infos 67/72 ≠ при идентичном коде. Проверка: построчный diff `extract_log` vs bot-3 extract.log+extract207 по n/poly_pos/finds_pos/n_gt_*; hash ObjectPoly/FindsPoints + MagGr серый для 207–212.
2. **Нет замороженного снимка данных/npz эталона — byte-repro недостижим.** Док-во: два retrain (d173a75, b0add08) дают один n/pos ≠ REF; MODELS из bot-3 28.10 без архива feats. Проверка: поиск архива `pixgbm/feats/*.npz` / хэшей растров на PC — не найдено.
3. **pick_rasters 1 vs 4 (hp_energy) — ОТКЛОНЕНА как причина Δp.** Default gray; код pick идентичен 441bb71; оба берут серый + цвет_2 как отдельный ключ.
4. **LGBM params / `--match-reference` — НЕ первичная.** names+300 trees; seed=0 → bagging_seed 7719 как в REF parameters; d173a75 с deterministic тоже FAIL при том же n/pos.
5. **207 «уч7» vs «7» — не пропуск растра.** Fallback с 82d23a0; 207 есть в NEW и в REF extract207; лёгкий сдвиг pos 5473→5576.

### Выбор: **(b)** e2e-metric вместо byte-match
Точное sha/Δp≤1e-4 воспроизведение эталона **недостижимо** без снимка данных/npz 28.10. Код признаков совпадает; расходится обучающая матрица на текущем `F:\2\train-expds-3`.

**Предлагаемый критерий приёмки (вместо sha / max\|Δp\|≤1e-4):**
- Конфиг: `[geophys.pixgbm] enabled=true, replace=true` (вариант **S**, как ACCEPTED→v2, REPORT7), остальные новые флаги OFF.
- Сравнить два прогона одного пайплайна: **REF** `models/pixgbm/all/{poly,finds}.txt` vs **NEW** из `pc_jobs/bot6_pixgbm_b0add08/models/all/`.
- Наборы: (1) **mock e2e** `F:\expds3\mock_val` — ALL_F1 и слои; (2) **honest-10** (`scripts/scoreboard.py --honest 10`) в Docker `pantum/n7q:20261024`.
- Pass: Δ(F1_new − F1_REF) **≥ −0.01** на mock ALL и на honest-10; **ни один слой / класс не падает > 0.01** (PROTOCOL §ворота); флаг в default.toml остаётся OFF до merge.
- Источник Δ=0.01 — тот же порог «no-drop > 0.01» PROTOCOL / VARIANTS_DECLARED; не выдуман из воздуха. Эталонный выигрыш S vs stack был +0.0099…+0.0141 (bot-3 honest_lofo) — здесь сравнение **веса vs веса**, не vs стека.

**Не делать новый full retrain** ради sha. Использовать уже полученные NEW-веса @b0add08.

**Запрос в очередь:** `pc/QUEUE.md` → READY-TO-LAUNCH GPU e2e-metric REF vs NEW @b0add08; runner `pc/runners/run_bot6_pixgbm_e2e_metric_b0add08.ps1`.
| Поле | Значение |
|---|---|
| Путь | **(b)** |
| SHA | `b0add08e194853a351d520df19cfb9810c4db983` (без нового коммита) |
| Команда | runner выше; ядро: docker `pantum/n7q:20261024` ×2 (REF/NEW model_dir) + `e2e_score.py` |
| Критерий | ΔALL(F1_NEW−F1_REF) ≥ **−0.01**; no layer drop **>0.01**; STATUS=DONE |
| JobDir | `C:\Sloi\pc_jobs\bot6_pixgbm_e2e_metric_b0add08` |


---

---

## PC @3f61ea8 — краш pick_rasters (отрицательный, 29.10 ~17:34 МСК) → фикс b0add08

| Поле | Значение |
|---|---|
| SHA краша | `3f61ea8f337371a26db25163e34526f4f3b6e80d` |
| Ошибка | `TypeError: pick_rasters() takes 1 positional argument but 4 were given` @ `scripts/pixgbm.py:148` `cmd_extract` |
| Логи | `C:\Sloi\pc_jobs\bot6_pixgbm_3f61ea8\retrain.err` |
| **Корневая причина** | Коммит `80d50b88` перевёл 3 call-site на 4-arg API `pick_rasters(files, rule, mt, crs)` (hp_energy). Cleanroom vendored `expds/geo_pixgbm.py` @sha256 `35e4331bbc690e7eb3b31e07cc290837ce9fd96a05c92f4c999f24eceafda35b` — сигнатура 1-arg `pick_rasters(files)` как в эталоне обучения @`441bb71`. **Не** регрессия Cyrillic-safe save/load и **не** `--match-reference`. |
| Grep сигнатур tip vs 441bb71 / geo | Единственный ломающий drift: `pick_rasters` 4↔1. Остальное: `cmd_oof`+`poly_components` есть в geo 35e4331b; `save_booster`/`load_booster` — локально в scripts; `load_map`/`label_masks`/`features_for`/`prob_map`/`poly_candidates`/`finds_candidates` совпадают по arity. |
| Фикс SHA | **`b0add08e194853a351d520df19cfb9810c4db983`** — три вызова → `G.pick_rasters(reg.rasters.get(mt, []))`; тест `tests/test_pixgbm_cmd_extract_e2e.py` (синтетическое дерево как F:/2/train-expds-3). |
| Тесты | 38 passed (полная bot-6 + gate pixgbm); peaks×2 pre-existing. |
| PC следующий | runner `pc/runners/run_bot6_pixgbm_b0add08.ps1`; threads=2; `--match-reference`; root `F:\2\train-expds-3`. |

**Точная команда:**
```powershell
$env:KOZ3_OUT = "C:\Sloi\pc_jobs\bot6_pixgbm_b0add08"
$env:PIXGBM_NUM_THREADS = "2"
# wt @ b0add08e194853a351d520df19cfb9810c4db983, geo_pixgbm.py sha 35e4331b…
python scripts\pixgbm_retrain_cleanroom.py --enable-retrain --match-reference --root "F:\2\train-expds-3" --out $env:KOZ3_OUT --rounds 300 --num-threads 2 --metric-tol 1e-4 --expected-hashes "models\pixgbm\all\MODELS_NOT_IN_GIT.md"
```

---

## Kaggle v6 — отрицательный результат (изм., 29.10 ~16:40 МСК)

**Не переделываем ядро.** Pull: `/workspace/kaggle/pulls/koz3-bot6-pixgbm-cleanroom` @ commit `b40ffd4`.

| Поле | Значение (из артефактов) |
|---|---|
| STATUS.txt | `DONE` `2026-10-29T13:37:40Z` `wall_s=8.3` |
| after_train | `returncode=0` `threads=4` |
| train wall_sec | `0.1` (`retrain_log.json`) |
| models | **нет** `poly.txt` / `finds.txt` (`compare.sha256` = null; metric checks `missing new`) |
| feats/extract_log.json | `[]` |
| models/train_info.json | `{}` |
| rebuild s000 | `members_geo=224` (small_files labels); shard step `error=no_blob_mappings_in_manifest`, linked=0 |
| rebuild s001–s009 | linked=0, copied=0; **skipped_non_geo** = 37,13,41,13,8,14,22,5,22 (сумма 175); mapped = те же non-geo |
| geo_region_dirs | 18 имён 200_–217_ (каталоги из labels), **растров geo нет** |

**Причина:** манифесты xfer s001–s009 не содержат blob-mapping для растров geo 200–217; в s000 только small_files (разметка). Обучение на пустом дереве → 0 моделей за ~8 с.

**Ложный exit 0:** `metric_fallback.status` становился `ok` при всех checks=`missing new` (пустое `all(within_tol)`). Исправлено в `d173a75` (+ Windows `resource` в `scripts/pixgbm.py`).

**Следующий шаг:** PC retrain на `F:\2\train-expds-3`, out `C:\Sloi\pc_jobs\bot6_pixgbm`, commit **`d173a75`**, см. `pc/QUEUE.md`.

Строка EXPERIMENTS.csv: `2026-10-29 16:42,bot-6,…,fail_data`.


## Данные Kaggle (открытый вопрос закрыт)

**Единого датасета train geo 200–217 нет.** Источники:

| Dataset | Содержимое |
|---|---|
| `dmitrijkv/koz3-xfer-s000` | `small_files.tar.zst` + manifest (разметка/UTM/extra; geo 200–217 labels в small) |
| `dmitrijkv/koz3-xfer-s001`…`s009` | крупные `f####.tif`/`.laz`; mapping в manifest шарда |
| `dmitrijkv/koz3-gt-train-v2` | consolidated GT (overlay; PIXGBM в основном берёт ObjectPoly/FindsPoints из s000) |

Ядро: `rebuild_train_tree.py` → `/kaggle/working/train-expds-3` (только префиксы 200–217), затем train. Offline s000: `/home/box/shared/koz3/kaggle/koz3-xfer-s000_manifest.json`.

**Манифесты, которые ещё нужны от bot-1 (чтобы сузить attach):** `manifest.json` шардов **s001, s002, s003, s004, s005, s006, s007, s008, s009**. До них в metadata — все s000–s009.

---

## Статус
- Tip: **`b0add08e194853a351d520df19cfb9810c4db983`**; PC retrain DONE FAIL sha/Δp; путь **(b)** e2e-metric. PR #5 draft.
- Тесты @b0add08: **38 passed** (bot-6); bot-1 full **674 passed** (bot-6 suite + pixgbm e2e/retrain/xfer/threads/maps/bot6_*). `test_geo_pixgbm_peaks` 2 fail — pre-existing `mode=` vs REF geo 35e4331b (не этот фикс). `box/run_tests.sh` stale `test_check_compliance.py` — не трогали.

## Findings (кратко)
- Обучение: `scripts/pixgbm.py` / clean-room `pixgbm_retrain_cleanroom.py`; lightgbm==4.7.0.
- Детерминизм (box): threads=1 synthetic identical; threads1≠threads8.
- Провенанс: КОЗ-3 geo 200–217; EN-линия кода признаков; clean-room → новое обучение.
- SIGTERM: 67284ab/af97af0 NOT fixed; eb268d6/v3 FIXED (`make_pool`).

## Команда / env
`PIXGBM_NUM_THREADS` (default cpu_count), `PIXGBM_ROUNDS=300`, `PIXGBM_METRIC_TOL=1e-4`.  
Ядро само: rebuild → `python scripts/pixgbm_retrain_cleanroom.py --enable-retrain --root $TREE --out … --num-threads $N`.

## 2026-10-29 19:55 МСК — e2e-metric r1 INVALID, r2 queued
- r1 (@b0add08, pantum/n7q:20261024): REF и NEW дали одинаковые ALL F1 0.312509, Δ 0 по всем слоям — **ложное равенство**. `expds/geo_pixgbm.py` стоит в `.gitignore` и не отслеживается git, поэтому в PC-worktree его нет; `expds/pipeline.py` ловит ImportError и молча уходит в классику. Модели pixgbm в обоих прогонах не загружались.
- Побочный вывод: любой прогон из чистого git-checkout с `[geophys.pixgbm] enabled=true` без вендоринга модуля молча выключает PIXGBM. Стоит проверить, что сабмит-зипы v2/v3 содержат `expds/geo_pixgbm.py` (sha 35e4331b для v2 eb268d6), и что mock-база 0.3125 считалась с ним.
- Гейт r1 упал с `could_not_parse_delta_ALL`: парсер не знал формат `e2e_score.py` (`rows`). В r2 исправлено.
- r2: `pc/runners/run_bot6_pixgbm_e2e_metric_b0add08_r2.ps1`. Вендорит модуль с проверкой sha, логирует sha моделей внутри контейнера, падает, если pix не отработал, и записывает, совпали ли geojson побайтно.

---

## r3 — PIXGBM REF vs NEW on train geo 200–217 (29.10 ~20:25 МСК)

Mock r1/r2 **CANCELLED** (no maggr/etgr/terfl → PIXGBM no-op). r3 targets real train geo regions.

| Поле | Значение |
|---|---|
| SHA | `13082a3e1f14be1590593f9fb887641cccb0484c` (STRICT module-missing guard) |
| JobDir | `C:\Sloi\pc_jobs\bot6_pixgbm_geo_r3` |
| Wt | `C:\Sloi\wt_bot6_pixgbm_geo_r3` |
| Regions | selective copy → `JobDir\regions` from `F:\2\train-expds-3` (RO) |
| GT | `bot6_r3_build_gt.py` → `JobDir\gt.geojson` (maggr/etgr/terfl only) |
| Vendor | `geo_pixgbm.py` sha `35e4331bbc690e7eb3b31e07cc290837ce9fd96a05c92f4c999f24eceafda35b` (not committed) |
| REF | archive zip_v2 `models/pixgbm/all` (poly `d1fe3efa…`, finds `5fa8c09d…`) |
| NEW | `C:\Sloi\pc_jobs\bot6_pixgbm_b0add08\models\all` |
| Overrides | `EXPDS_PIXGBM_STRICT=1`, `PIXGBM_BUDGET_S=100000`, `EXPDS_SOFT_CEILING_S=0`; toml `[time] budget_s=100000 watchdog=false` |
| Gate | `bot6_r3_gate.py`: INVALID 10/11/12/13/14/19; PASS iff \|ΔF1\|≤0.01 on ALL **and** maggr/etgr/terfl |
| Disk | C: only for copy (abort exit **15** if plan >10 GB); **no F: writes**; expected copy ~**3.5 GB** (geo200 pack ref; manifests lack 200–217 — «1» verifies plan) |
| Runner | `pc/runners/run_bot6_pixgbm_geo_r3.ps1` + `launch_bot6_pixgbm_geo_r3_schtasks.ps1` |
| QUEUE | entry above RETRY bot2 A3 |

### Self-check (box)
- `bot6_r3_gate.py --self-check` → pass (empty→11, pass→0, fail_eq→9, identical→14)
- `bot6_r3_select_copy.py --self-check` → pass (excludes LAS; exceed→15)
- `bot6_r3_build_gt.py --self-check` → pass (drops non-geo markup)

### PC uncertainties for «1»
- Exact free space on C:/F: at launch time; plan GB after `select_copy` (must log before copy).
- Whether all of 200–217 exist under `F:\2\train-expds-3` with maggr/etgr/terfl (else discover-by-raster).
- REF model path if archive layout moved; NEW must exist at b0add08 JobDir.
- Docker wall time for 18 geo regions with budget 100000 / soft ceiling off — long GPU slot.

---

## 2026-11-01 — steering 9b411c4 / PR #20 review + pixgbm-cr package (Europe/Moscow)

### (a) Verify claims @ `9b411c4`

| Check | Result |
|---|---|
| Fast-forward worktree to `9b411c4` without overwrite | OK |
| `expds/geo_pixgbm.py` committed? | **No** — still gitignored (`.gitignore:74`); disk copy sha256 `35e4331b…` matches MODELS vendor pin |
| pytest | **15 passed** in 15.11s (geo present on disk; PR text said 10+1 skip) |
| `scripts/repro_pixgbm_sigterm.py` | exit 0; hang on `67284ab`/`8594842`; clean on `eb268d6`/`36af6b7`/`78af57f`; `af97af0` no fork pool — matches claim |
| smoke retrain | `SMOKE_IDENTICAL` |
| Artifacts | `/tmp/bot6_pytest.txt`, `/tmp/pixgbm_sigterm.json`, `/tmp/pixgbm_smoke.txt` |

### (b) Clean-room conflict vs «1» goal

- `9b411c4` / PR #20 **reuses EN-lineage** `geo_pixgbm.py` (vendored at runtime) + retrains `models/pixgbm/all` → **P-retrain reproduction**, **not** clean-room under `docs/PIXGBM_DECISION.md` / SPEC (independent features + new `models/pix_clean`).
- **«1» actual goal:** main tip builds geophysics no worse than v2, `|dF1|≤0.01` on 200–217, run on **Kaggle CPU or box (no PC)**.
- **Conflict:** treating PR #20 as the clean-room answer would fail the OWNER rule and the decision doc. Tip already has true clean-room code: `expds/pix_clean.py` behind `[geophys.pixgbm] impl="clean"` (default `enabled=false`). Orch STICKY: clean pack P1 parity PASS / P4 zip built on Drive — LB only on Elena «да».
- Equivalence **not executed on this box**: no local MagGr/EtGr/TerFl for 200–217 (xfer hollow; need `koz3-w4-geo-s000/s001`). r3 READY runner never launched (no `pc/results/bot6_pixgbm_geo_r3`).

### (c) Main tip SIGTERM fix?

| Path | Status |
|---|---|
| Committed tip (`origin/main` = `f127db8`) | **No** `make_pool` / `_pool_init` in tree (module gitignored) |
| Vendored `geo_pixgbm.py` sha `35e4331b…` (v2 build-time) | **Yes** — `make_pool` + `_pool_init` (SIG_DFL + pthread_sigmask) as in `eb268d6` |
| `expds/pix_clean.py` (tip clean path) | **N/A hang** — uses threads, not fork `Pool`; no inherit-of-`arm_signal` hang |
| Tip test | `tests/test_bot6_pix_pool_sigterm.py` skips if module absent; passes when vendored |

### (d) Package

Ready (not pushed): `/workspace/kaggle/push/pixgbm-cr/`

- `kernel-metadata.json` — `enable_gpu=false`, private, datasets w4-geo-s000/s001 + xfer-s000 + gt-train-v2
- `pixgbm_cr_gate.py` — GATE_MODE=zip|tip|score; rebuild via `rebuild_w4_geo_tree.py`; applies `helpers/bot6_r3_gate.py`
- README + `notes/SPEC.json` document REF=v2 vs CAND=`impl=clean`, and the 9b411c4 conflict
- Attach REF/CAND zips or models before «1» pushes

### Deliverables summary for «1»

1. **Clean-room meaning:** independent rewrite (`pix_clean` / SPEC), not EN paste. **Impl on tip:** `expds/pix_clean.py`, flag `impl="clean"`, models `models/pix_clean`, OFF by default. Tooling SHA `9b411c4` = retrain helper only.
2. **Equivalence 200–217:** **not run here** — readiness = package + datasets; orch P1 PASS is separate evidence on Drive pack.
3. **SIGTERM:** cause = child inherits `watchdog.arm_signal`, `pool.terminate()` → hang; fix = block SIGTERM in parent during fork + SIG_DFL/unblock in `_pool_init`; verified by `repro_pixgbm_sigterm.py`; tip clean path unaffected.
4. **Package path:** `/workspace/kaggle/push/pixgbm-cr/` (do not push from bot-6).
5. **rerank_arch:** default OFF in `configs/default.toml`; historically W5/orch peak-rerank — **not assigned to bot-6**; do not implement.

### Blockers / questions for «1»

- Attach private REF (v2 zip) + CAND (clean-pack zip or `models/pix_clean`) to the kernel before push.
- Confirm whether «1» wants this gate vs orch Drive P4 pack as the authoritative tip≈v2 answer.
- Kaggle API keys `KAGGLE_KEY_1`/`KEY_2` unset on this box session (only short `KAGGLE_API_TOKEN`); bot-6 did not push.

## 2026-11-01 — «1» review fix: REF/CAND datasets + internet=false (Europe/Moscow)

Package `/workspace/kaggle/push/pixgbm-cr/` updated; **still not pushed**.

- `enable_internet=false`, `enable_gpu=false`
- kernel id: `dmitrijkv/koz3-bot6-pixgbm-cr-gate`
- dataset_sources now include REF/CAND/wheels:
  `dmitrijkv/koz3-pixgbm-cr-ref`, `…-cand`, `…-wheels` (+ w4-geo-s000/s001, xfer-s000, gt-train-v2)
- Upload-ready `ds/` folders staged on box (weights + `geo_pixgbm.py` **not** in git)
- Existing `elenakonkova/koz3-e2e-baseline-v2` noted but 403 from dmitrijkv token → tip REF staged locally
- CAND clean-pack **not** on box — «1» must drop Drive zip into `ds/pixgbm-cr-cand/` before create
- Vendored: zstandard 0.25.0 cp310/311/312 manylinux2014_x86_64 only; no network pip
- `python pixgbm_cr_gate.py --syntax` → DRY_RUN_OK
- w4-geo-s000/s001 still 403 for this token — «1» must confirm ACL/existence


## 2026-11-01 21:20 — Kaggle 200–217 equivalence gate (REF v2 eb268d6+FR1h-off vs CAND P4 clean e1g1)

Scored on box from Kaggle outputs (e2e_score + compute_metrics, GT 4561 features).

| markup | GT | F1 ref | F1 cand | Δ |
|---|---|---|---|---|
| ALL | 4561 | 0.637 | 0.685 | +0.048 |
| etgr | 658 | 0.543 | 0.684 | +0.142 |
| maggr | 3726 | 0.603 | 0.586 | −0.017 |
| terfl | 177 | 0.000 | 0.000 | 0 |

Verdict: **FAIL equivalence** (|ΔF1| > 0.01). Not a drop-in for v2. maggr is worse on 200–203 and 207–211 (−0.02…−0.34, 201 −0.337); etgr/Гнёздово 212–217 better.
Gate exit flagged INVALID for cand only because the clean impl does not print the v2 step/phase log lines (log-parser artifact); the cand run produced all 18 regions.
