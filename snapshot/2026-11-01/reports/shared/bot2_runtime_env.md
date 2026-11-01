# Бот 2 — среда рантайма LB vs ПК (host Windows vs Linux Docker) — 29.10.2026 ~17:35 МСК

| Поле | Значение |
|---|---|
| **Цель** | С evidencе: где реально крутится LB; чем host Windows venv отличается от Linux Docker; как считали v2 baseline 0.3125; нужен ли пересчёт ворот в Docker. |
| **Гипотеза** | LB = Linux Docker организаторов (`pantum/n7q`). Vendor D-FINE manylinux → на Windows host молча/громко недоступен. База mock 0.3125 из kit `v3e2e` была Docker; поздние A/B/A2 — host `.venv`. Для ворот с D-FINE ON нужен единый Docker-раннер. |
| **Критерий приёмки** | Отчёт с путями/цитатами; список расходящихся компонентов; вердикт по пересчёту; при YES — заявка QUEUE «[bot2] v2 base в docker». |
| **Бюджет** | ~0,5 ч CPU box (только чтение + QUEUE/отчёт). PC GPU: после A3. |
| **Ветка** | `bots/bot2-e2e-v3-prep` @ **`829ea7d`** (loud UNAVAILABLE + A3 docker runner). |

Связанные: `reports/bot2_dfine_localization.md`, `reports/bot2_v3_hotfix.md`, `pc/runners/run_bot2_a3_dfine_docker_import.ps1`, BOARD §«Факты e2e v3».

---

## 1. Где исполняется LB-посылка

**Вердикт: Linux Docker организаторов**, образ из `metadata.json`, entrypoint как в архиве.

| Источник | Цитата / факт |
|---|---|
| `metadata.json` | `"image": "pantum/n7q:20261024"`, `"entry_point": "python inference/main.py"` |
| `docker/Dockerfile` | `FROM python:3.12-slim-bookworm`; `torch==2.11.0` / `cu128`; «Окружение для запуска решения на платформе КОЗ №3»; «В образе ТОЛЬКО окружение: без кода и без весов»; «PyTorch CUDA 12.8: RTX 4090» |
| `docs/ARCHITECTURE.md` | «На площадке видеокарта RTX 4090… Контейнеру доступны 96 ГБ, 24 ядра, `/dev/shm` 16 ГБ»; «Образ Docker — окружение без кода и весов (`docker/Dockerfile`, torch 2.11.0, cu128). В `metadata.json` стоит `pantum/n7q:20261024`» |
| `docs/PLAN.md` | «Образ опубликован: `pantum/n7q:20261024`»; лимиты контейнера; прогоны в `expds-runtime:…` / `pantum/n7q` |
| `docs/LB_LOG.md` | Все квалификации с образом `pantum/n7q:20261024` (или `…20261023d`); пример: «Образ `pantum/n7q:20261024`» |
| `README.md` | «Для контейнера платформы Rust собирается так: `python scripts/build_rust.py linux --docker`. … Имя опубликованного образа — в `metadata.json`» |
| Organizer materials `docs/background/KOZ3_ORGANIZER_MATERIALS_2026-10-21.md` | «до 96 ГБ контейнеру»; «образ публичный на Docker Hub»; «Python ≥ 3.12»; шаги «Загрузка образа» |
| `scripts/platform_sim.ps1` | «лимиты: 96 ГБ памяти, 8192 процесса и потока, /dev/shm 16 ГБ» |

`docs/RULES.md` в этой ветке **нет** (файл отсутствует); правила площадки закрыты материалами + ARCHITECTURE/PLAN/LB_LOG.

**LB vs mock:** PROTOCOL §5 — ворота к базе **v2 mock 0,3125 / LB 0,41238**. LB `eb268d6` ≈ **0.4123779189** (`reports/bot7_honest_eval_layer.md`). Mock 0.3125 — локальный 2-участковый GT, не public val.

---

## 2. Использовал ли v2 D-FINE / Linux-only vendor?

**Нет.**

| Факт | Evidence |
|---|---|
| `[sp_dfine] enabled = false` в `configs/default.toml` | строки ~456–464; комментарий «В архиве по умолчанию ВЫКЛ.» |
| `[gor_gate] enabled = false` | ~477–478 |
| `scripts/build_submission.py` | vendor кладётся **только** если `sp_dfine` или `gor_gate` enabled; иначе zip без `inference/lib/vendor` |
| Zip v2 kit | `submission_bot7cand_v2_eb268d6.zip` — база без make_on C |

Следствие: **v2 baseline не зависит от manylinux-vendor transformers**. Отсутствие D-FINE на Windows host **не** объясняет число 0.3125 само по себе (D-FINE был OFF). Риск host≠Docker для v2 — другие компоненты (accel, affinity, float/CUDA path), не vendor.

---

## 3. Как считали mock 0.3125 (v2 baseline)

| Артефакт | Среда | Evidence |
|---|---|---|
| **Первичный kit `v3e2e` → `out_v3fix2/res_v2` ALL≈0.3125** | **Docker**, GTX 1650 | BOARD: «Факты e2e v3 на ПК (29.10 14:43–14:48, zip v3on_0876090, **docker**, GTX 1650) … F1 ALL **v2 0,3125** → v3 0,2880» |
| Числа ворот | ALL **0.312509**, li 0.336983, sp_or 0.242915, ae 0.095556 | `reports/bot2_v3_hotfix.md`, `reports/bot5_v4.md` |
| Zip | `C:\Sloi\pc_jobs\v3e2e\kit\submission_bot7cand_v2_eb268d6.zip` | runners A2/A3/hotfix |
| GeoJSON базы | `…\out_v3fix2\res_v2\result.geojson` (fallback `out\res_v2\`) | runners |
| **Hotfix A/B @1ec2d7c** (ALL **0.312873**) | **Host Windows `.venv`**, не Docker | `pc/runners/run_bot2_v3_hotfix_1ec2d7c.ps1`: `$Py = '…\.venv\Scripts\python.exe'` (QUEUE просил docker — раннер сделал host) |
| **A2 @fb66c89** | Host `.venv` → ImportError `_regex` / n_raw=0 | тот же `$Py`; `reports/bot5_v4.md` §A2 |
| **bot4 large_objects mock** | Host Windows-11 | `main_on.log`: `среда: {'python': '3.12.10', 'platform': 'Windows-11-…', 'accel': 'rust 0.1.0…'}` |
| **A3 @829ea7d** | Docker `pantum/n7q:20261024` | `run_bot2_a3_dfine_docker_import.ps1`; STATUS_LIVE RELAUNCHED 17:30 |

Итог: **каноническая база 0.3125 = Docker kit**. Поздние «≈v2» на host дали **0.312873** (Δ≈+0.00036 к docker-базе) — уже **cross-env**. bot5 flags-OFF identity сверялся с этими числами; bot5 сам отметил риск host без vendor (`reports/bot5_v4.md`).

---

## 4. Компоненты: Windows host venv vs Linux Docker

| Компонент | Docker (LB / pantum/n7q) | Windows host `.venv` | Поведение при расхождении |
|---|---|---|---|
| **ОС / Python** | Linux, Python 3.12 (slim-bookworm) | Windows-11, 3.12.10 (лог bot4) | — |
| **Образ / CUDA** | torch 2.11.0+cu128; LB: RTX 4090; PC Docker: GTX 1650 | тот же torch в venv; GTX 1650 | TF32/FP32 политики как в README; пороги снимались в FP32 |
| **`inference/lib/vendor/**`** (transformers 5.21.0, regex, tokenizers, …) manylinux **cp312** | OK | **FAIL** load `.so` (`_regex` и др.) | До 829ea7d: тихий пустой D-FINE → A≡B. С 829ea7d: **SP_DFINE_UNAVAILABLE** + assert в `make_on`. Влияет только при `sp_dfine`/`gor_gate` ON |
| **`inference/lib/expds_core.abi3.so`** | manylinux abi3 (`scripts/build_rust.py linux [--docker]`) | ImportError → **NumPy** *или* отдельная win-сборка на ПК (bot4 log: `ускоритель: rust 0.1.0`) | `accel.py`: «одинаковый результат» rust/numpy (`tests/test_accel.py`); без `.so` — медленнее. `build_submission`: предупреждение «на платформе будет NumPy» |
| **lleaves / treelite `.so`** | в архив **не** кладутся (default `backend=lightgbm`) | то же | при `lleaves` — silent fallback на lightgbm (`geo_pixgbm.py`) |
| **`sched_getaffinity`** (pixgbm procs) | Linux | нет → другой default потоков | conf comment: affinity Linux-only |
| **cgroup / mem / shm** | 96 ГБ / shm 16 ГБ на площадке; PC Docker часто cap 16 ГБ | host RAM | soft_ceiling / soft_skip фаз |
| **D-FINE import path** | vendor → pin | vendor fail → env (часто без transformers) → UNAVAILABLE | `sp_dfine.import_dfine`; `vendor_platform_hint` в 829ea7d |
| **gor_gate / DINOv2** | нужен тот же vendor | host: `No module named 'transformers'` (bot5 train FAILED) | только при enabled |

Молчаливая деградация (до loud-fix): пустой D-FINE, accel→numpy, lleaves→lightgbm, soft_ok skip фазы.

---

## 5. Вердикт: нужен ли пересчёт gate baseline в Docker?

**ДА — пересчитать / зафиксировать v2 base (+ honest-10) тем же Docker-раннером, что A3.**

Почему, несмотря на то что kit 0.3125 уже был Docker:

1. Поздние воротные сравнения (hotfix A/B 0.312873, bot5 flags-OFF) шли на **host** и смешали среды.
2. Любой gate с **D-FINE / gor_gate / vendor ON** обязан быть Docker; база сравнения должна быть из **того же** образа `pantum/n7q:20261024` и того же runner-шаблона.
3. PROTOCOL §5 требует mock **и** honest-10; honest-10 docker-reference для v2 zip в shared не зафиксирован рядом с A3.
4. Нужен явный лог без `SP_DFINE_UNAVAILABLE` (v2 OFF) и послойный F1 vs 0.3125/0.3129.

Ожидание: mock ALL ≈ **0.3125** (как BOARD docker), не 0.312873 host. Если Docker снова ≈0.3125 — host-ε документируем и **каноном** остаётся Docker. Если Docker ≠0.3125 — обновить ворота.

---

## 6. QUEUE

Добавлено: **`[bot2] v2 base в docker`** в `pc/QUEUE.md` (после A3). Runner: модель `run_bot2_a3_dfine_docker_import.ps1`, zip v2 `eb268d6`, без overlay D-FINE ON.

---

## 7. Следующий шаг

1. Дождаться A3 @829ea7d (n_raw>0 в Docker).  
2. Прогон QUEUE v2 base Docker → mock ALL/layers + honest-10.  
3. Записать EXPERIMENTS; при Δ к 0.3125 — обновить PROTOCOL/BOARD.  
4. Все дальнейшие e2e с vendor — **только** Docker (не host `.venv`).
