# Анализ бота «Слои»: бесплатные GPU и ферма аккаунтов (КОЗ №3)

> Собрано 2026-10-15 ~21:12 МСК. Источник: `/workspace/gpu-orch/`.
> Организаторы дали добро на зарубежные платформы. Оплаты нет — только free + GTX 1650.

## Оглавление

1. [PLAN_FREE_GPU_ORCH](#plan_free_gpu_orch)
2. [OSINT_FREE_GPU_WORLD](#osint_free_gpu_world)
3. [FARM_DESIGN](#farm_design)
4. [ACCOUNT_POOL](#account_pool)
5. [RISKS_TOS](#risks_tos)
6. [NEXT_ACTIONS](#next_actions)
7. [pool.example.yaml](#poolexampleyaml)


---

<a id="plan-free-gpu-orch"></a>

# План: оркестрация бесплатных GPU (КОЗ №3 + боты)

> 15.10.2026. Пул — «Слои». Политика: **free-only + local 1650** (без immers/PAYG). Зарубежье разрешено оргами. См. также OSINT + FARM_DESIGN.

## 0. Главное
1. **Пул = по одному легальному аккаунту на человека** (5 участников команды). Ротация идёт между *людьми и провайдерами*, а не между фейковыми аккаунтами одного человека: несколько аккаунтов на одного нарушают ToS Kaggle/Colab и условия акций (см. RISKS_TOS.md).
2. **Зарубежные платформы:** организаторы **разрешили** Kaggle/Colab и аналоги (15.10.2026). Политика spend: **только free / гранты без карты или $0** — без immers/Cloud4Y/PAYG.
3. **Целевое железо финала — RTX 4090 24 ГБ, ≤30 мин валидация, архив ≤2 ГБ, DockerHub** (PDF консультации). **Бесплатного 4090 нет:** профилируем на Kaggle T4/P100 + локальный GTX 1650 и закладываем headroom; 1:1 гарантия 30 мин@4090 без железа **недостижима** (см. FARM_DESIGN §7, OSINT).
4. Боты Kontur/Mandat/AeroBIM GPU **не требуют** — в пул не включаем.

## 1. Бэкенды и роли
> Сверка: **2026-10-15 (MSK)**. Источники — ссылки в ячейках. Цифры без ЛК помечены `[проверить]`.

| Бэкенд | GPU / квота | Для чего | Ограничения / источник |
|---|---|---|---|
| Kaggle Notebooks | **P100** (док.) + опция **T4×2** в UI; **~30 ч GPU/нед** (иногда выше, floating); сброс **сб 00:00 UTC = сб 03:00 МСК**; сессия GPU до **~9 ч** (док. tips) | основное обучение U-Net, k-fold (5 человек × свои аккаунты) | 1 аккаунт/человек (ToS); телефон; данные: зарубежье разрешено оргами. Источники: [efficient-gpu-usage](https://www.kaggle.com/docs/efficient-gpu-usage), [GPU tips](https://www.kaggle.com/page/GPU-tips-and-tricks), [ToS](https://www.kaggle.com/terms) |
| Google Colab free | GPU **не гарантирован** (часто T4); **фиксированной ч/нед нет**; сессия ≤**12 ч** | короткие эксперименты, отладка | динамические лимиты, idle-обрывы. [Colab FAQ](https://research.google.com/colaboratory/faq.html) |
| Lightning.ai free | разово ~30 credits (5+**25 за карту**) | только если у участника уже есть карта и $0 charge ок; иначе **вне пула** | [pricing](https://api.lightning.ai/pricing/) |
| Modal Starter | $30/мес | **вне пула:** payment method on file обязателен ([billing](https://modal.com/docs/guide/billing)) | — |
| HF ZeroGPU | **не обучение**: Spaces; free = **5 GPU-мин/сутки**; железо: RTX Pro 6000 Blackwell (large 48ГБ / xlarge 96ГБ); hosting ≤2 Spaces (free, аккаунт ≥30 дн) | демо | Gradio only. [docs ZeroGPU](https://huggingface.co/docs/hub/spaces-zerogpu) |
| Cloud.ru free tier / бонусы | **GPU нельзя** на бонусах; для GPU ВМ нужно **≥15 000 ₽ cash** на балансе. Free: S3 **15 ГБ**/мес + Container Apps CPU (до 25 vCPU·ч / 50 GB·RAM·ч) — **без GPU** | CPU: RVT-каналы, тайлы/чекпойнты в S3 | [free tier](https://cloud.ru/offers/free-tier), [GPU docs](https://cloud.ru/docs/virtual-machines/ug/topics/concepts__gpu) |
| VK Cloud приветствие | **5 000 ₽ / 60 дн** (ФЛ); из них **2 000 ₽** на ВМ+бэкапы, 3 000 на прочее; excl. лицензии/маркетплейс/CDN. **Грант стартапов GPU явно исключает**. Покрывает ли приветственный бонус именно GPU-флейворы — **`[проверить в ЛК]`** (нужна заявка на квоту GPU) | РФ-контур для конфиденциальных данных (если GPU доступен за бонус/cash) | [bonus](https://cloud.vk.ru/bonus/), [startup](https://cloud.vk.ru/startup/) |
| Локальный GTX 1650 4 ГБ | — | препроцессинг, smoke, профилирование, мелкий batch | 4 ГБ VRAM |
| Paperspace Gradient Free | Free **M4000** (notebooks) | доп. free GPU при capacity | 6 ч autostop; 1 notebook; [DO pricing](https://docs.digitalocean.com/products/paperspace/pricing/) |
| CN-лабы (Baidu / ModelScope / OpenI / Tencent CS) | V100/T4/акции | опционально, если участник уже в экосистеме | см. OSINT_FREE_GPU_WORLD.md |

**Вне пула (политика $0 / no-card):** immers, Cloud4Y, Modal (карта обязательна), Lightning +25 за карту, AutoDL rental, hyperscaler GPU trial.

Мифы: Oracle Always Free — **без GPU**; GitHub Codespaces GPU — **deprecated**; AWS/GCP/Azure free ≠ GPU; Studio Lab — закрыт для новых. Подробности: `OSINT_FREE_GPU_WORLD.md`.

## 2. Архитектура
```
[Git репо + Docker-образ] --> [Очередь задач (jobs.yaml / SQLite на box)]
        |                               |
   [Диспетчер] -- выбирает бэкенд по: тип задачи, VRAM, остаток квоты, разрешён ли тип данных
        |--> Kaggle (kaggle kernels push, API-токен владельца аккаунта)
        |--> Colab (вручную владельцем; ссылка-ноутбук из репо)
        |--> Paperspace Free / CN-лабы (опционально)
        |--> local GTX 1650
        |--> Cloud.ru CPU/S3 (чекпойнты)
        v
[Артефакты: чекпойнты+метрики -> S3 (Cloud.ru) / HF private / MLflow] -> [Сводка в боте «Слои»]
```
- **Идемпотентные задачи:** каждая job = (git SHA, конфиг, seed, fold) → детерминированный id; чекпойнт каждые N мин в S3, при обрыве задача возобновляется на любом бэкенде с последнего чекпойнта.
- **Fail-over:** таймаут/обрыв/исчерпание квоты → статус `PREEMPTED` → в очередь со следующим свободным бэкендом того же класса данных (зарубежье разрешено оргами; всё равно минимизировать лишние ПДн).
- **Квоты:** таблица остатка часов по (человек, провайдер) обновляется после каждой сессии; диспетчер не планирует больше 80% остатка.
- **Секреты:** токен каждого участника хранит **сам участник** (личный kaggle.json / vault item); никакой общей «копилки паролей»; боты получают только запуск через CI/workflow, не сырые креды. Не делиться логинами — это тоже нарушение ToS.
- **Качество:** фиксированные версии (Docker/requirements lock), одинаковый seed, AMP fp16 на T4; group k-fold по регионам; метрики считать официальным compute_metrics.py; сравнивать эксперименты только на одной сетке валидации. Финальную скорость на 4090 — если появится бесплатный доступ; иначе экстраполяция с T4/P100/1650 + headroom (FARM_DESIGN §7).

## 3. Что на каком GPU
| Задача | Бэкенд |
|---|---|
| RVT-каналы, тайлинг, DTM | CPU: Cloud.ru / локально |
| U-Net на модальность, k-fold | Kaggle (параллельно у 5 человек = 5 fold'ов) |
| Абляции v2 (heatmap, GBDT-стекинг) | Kaggle / Paperspace Free / CN-лабы |
| Подбор порогов по классам | CPU |
| Финальный профайлинг ≤30 мин @4090 | **нет free 4090** — профиль на Kaggle/1650 + запас; 4090 только если грант/партнёр $0 |
| Демо | HF ZeroGPU / Spaces (публичные данные) |

## 4. Боты
- «Слои» — владелец пула: ведёт очередь, таблицу квот, health-check, сводку.
- Основной — код пайплайна/Docker, PR.
- Kontur/Mandat/AeroBIM — без изменений, GPU не используют.

## 5. Handoff / документы
- `OSINT_FREE_GPU_WORLD.md` — карта free GPU мира.
- `FARM_DESIGN.md` — пул согласных участников, ёмкость, compliance (не anti-ban), финал без оплаты.
- Организаторы зарубежье **разрешили**; paid fallback **снят**.
- Дальше: собрать `pool.yaml` на 5 человек (NEXT_ACTIONS); диспетчер — код Основного по команде.


---

<a id="osint-free-gpu-world"></a>

# OSINT: бесплатные GPU в мире (КОЗ №3)

> Сверка: **2026-10-15 (MSK)**. Цифры только из публичных источников; без ЛК — `[проверить]`.  
> Контекст: организаторы **разрешили** зарубежные платформы. Бюджет GPU: **только free / гранты / кредиты без карты или с $0 spend**. Платный immers/Cloud4Y/PAYG **вне скоупа**.

## 1. Сводная таблица

| Провайдер | Регион / доступ из РФ | GPU | Лимит сессия / период | Условия (телефон/карта/edu) | Мультиаккаунт ToS | Ссылка | Дата |
|---|---|---|---|---|---|---|---|
| **Kaggle Notebooks** | США (Google); из РФ обычно работает, телефон `[проверить]` | P100 (док.); T4×2 в UI | ~**30 ч GPU/нед** (floating↑); сессия GPU ~**9 ч**; сброс сб 00:00 UTC | аккаунт + **телефон**; карта не нужна | **запрещён** >1 аккаунт/человек | [GPU docs](https://www.kaggle.com/docs/efficient-gpu-usage), [ToS](https://www.kaggle.com/terms) | 15.10.2026 |
| **Google Colab free** | США; из РФ часто нужен обход DNS/VPN `[проверить локально]` | часто T4, **не гарант.** | фикс. ч/нед **нет**; сессия ≤**12 ч** | Google-аккаунт; карта не нужна для free | злоупотребление free → блок | [FAQ](https://research.google.com/colaboratory/faq.html) | 15.10.2026 |
| **Lightning.ai free** | США | T4/L4/… по credits | **разово** до ~30 credits (5+25 за карту), 12 мес; «~80 GPU-ч на старт» маркетингом | **карта** для +25 credits → **вне $0/no-card политики**, если карта обязательна | 1 человек / abuse free | [pricing](https://api.lightning.ai/pricing/) | 15.10.2026 |
| **Modal Starter** | США | T4…H100 PAYG | **$30/мес** included | док.: **payment method on file обязателен** → **не подходит** под no-card/$0 | workspace seats | [billing](https://modal.com/docs/guide/billing), [pricing](https://modal.com/pricing) | 15.10.2026 |
| **HF ZeroGPU** | глобально | RTX Pro 6000 Blackwell (large/xlarge) | free **5 мин/сутки**; Spaces only | HF-аккаунт; PRO дороже | 1 аккаунт | [ZeroGPU](https://huggingface.co/docs/hub/spaces-zerogpu) | 15.10.2026 |
| **GitHub Codespaces GPU** | — | **нет** | GPU machine type **deprecated** (авг 2025); free = CPU core-hours | GitHub Free 120 core-h/мес (CPU) | — | [changelog](https://github.blog/changelog/2025-08-01-upcoming-deprecation-of-gpu-machine-type-in-codespaces/), [billing](https://docs.github.com/en/billing/concepts/product-billing/github-codespaces) | 15.10.2026 |
| **Oracle Always Free** | глобально | **GPU нет** | только AMD Micro + Ampere A1 CPU | карта часто нужна при регистрации `[проверить]` | — | [Always Free](https://docs.oracle.com/en-us/iaas/Content/FreeTier/freetier_topic-Always_Free_Resources.htm) | 15.10.2026 |
| **AWS Free Tier / credits** | глобально | **GPU в free plan нет** | Studio Lab: GPU ≤4 ч/раз / 4 ч/24ч — но **закрыт для новых** | карта; Studio Lab → миграция в платный Studio | — | [Studio Lab change](https://docs.aws.amazon.com/sagemaker/latest/dg/studio-lab-availability-change.html), [overview](https://docs.aws.amazon.com/sagemaker/latest/dg/studio-lab-overview.html) | 15.10.2026 |
| **GCP Free Trial $300** | глобально | **GPU нельзя** на trial | e2-micro CPU free tier; trial **запрещает** Add GPUs | карта | — | [free features](https://docs.cloud.google.com/free/docs/free-cloud-features) | 15.10.2026 |
| **Azure trial $200** | глобально | GPU SKU часто quota=0 на trial `[проверить]` | 30 дн credit | карта | — | Azure NC docs / trial | 15.10.2026 |
| **Paperspace Gradient Free** | DigitalOcean; доступ из РФ `[проверить]` | Free **M4000** (notebooks) | auto-shutdown **6 ч**; 1 free notebook; capacity не гарант. | Free plan; карта для апгрейда | 1 аккаунт | [pricing](https://docs.digitalocean.com/products/paperspace/pricing/), [features](https://docs.digitalocean.com/products/paperspace/machines/details/features/) | 15.10.2026 |
| **Cloud.ru free** | РФ | **GPU нельзя** на бонусах; GPU ≥15k ₽ cash | S3 15 ГБ; Container Apps CPU | РФ | 1 лицо / акция | [free](https://cloud.ru/offers/free-tier), [GPU](https://cloud.ru/docs/virtual-machines/ug/topics/concepts__gpu) | 15.10.2026 |
| **VK Cloud бонус** | РФ | GPU: грант стартапов **исключает**; welcome 5k — GPU в ЛК `[проверить]` | 5k ₽ / 60 дн (ФЛ), 2k на ВМ | телефон/документы | 1 акция/лицо | [bonus](https://cloud.vk.ru/bonus/), [startup](https://cloud.vk.ru/startup/) | 15.10.2026 |
| **Tencent Cloud Studio** | КНР; из РФ `[проверить]` | T4 (GPU workspace) | **2 машинных часа/день** (срок 7 дн); T4 жжёт **1.2** маш.ч за 1 ч ≈ **~1.67 ч T4/день** | QQ/WeChat/телефон CN часто | `[проверить ToS]` | [ежедневные часы](https://cloud.tencent.com/document/product/1039/131775) | 15.10.2026 |
| **Baidu AI Studio** | КНР/РФ-дружелюбно | V100 16/32 ГБ | **8 точек/день** ≈ 16 ч V100-16GB или 8 ч V100-32GB (офиц. справка) | телефон РФ/CN; экосистема **Paddle** | `[проверить]` | [aistudio](https://aistudio.baidu.com/), справка ai-doc AISTUDIO | 15.10.2026 |
| **ModelScope Notebook** | КНР | NVIDIA ~24 ГБ; AMD ~192 ГБ (акция) | NVIDIA: лимит в UI `[проверить]`; AMD ~**100 ч** через AMD Developer | китайский аккаунт; AMD-регистрация | `[проверить]` | [modelscope.cn](https://modelscope.cn/) | 15.10.2026 |
| **OpenI / 启智** | КНР | GPU/NPU по заявке | бесплатно/баллы через **заявки и акции**; фикс. квота не опубликована | регистрация + заявка | `[проверить]` | [openi.pcl.ac.cn](https://openi.pcl.ac.cn/) | 15.10.2026 |
| **AutoDL** | КНР | 3090/4090/… | **не free tier**: rental + разовый ваучер ~¥10 `[проверить]` | телефон CN | — | autodl.com | 15.10.2026 |
| **阿里天池 / PAI trial** | КНР | V100/P100/T4 в лабах | фикс. публичной квоты нет; research/edu vouchers `[проверить]` | Alipay/телефон | `[проверить]` | [tianchi](https://tianchi.aliyun.com/) | 15.10.2026 |
| **University / research credits** | по программе | любые GPU облака | Google Research Credits (faculty/PhD), AWS Cloud Credit for Research — **заявка**, недели weeks | edu-аффилиация; часто карта org | N/A | [Google research](https://edu.google.com/programs/credits/research/), [AWS research](https://aws.amazon.com/government-education/research-and-technical-computing/cloud-credit-for-research/) | 15.10.2026 |
| **Discord / referral promos** | разное | разовые $ | **нет** стабильного free GPU signup у RunPod/Vast (проверено косвенно); челлендж-кредиты ad-hoc | — | — | RunPod Discord и т.п. | 15.10.2026 |
| **Локальный GTX 1650** | дом/лаб | 1650 4 ГБ | безлимит по электричеству | — | — | железо команды | 15.10.2026 |

### Мифы (явно)
- **«AWS/GCP/Azure free = бесплатный GPU»** — ложь для trial/free tier (GCP trial прямо запрещает Add GPUs; AWS free plan GPU не включает; Azure trial GPU часто quota 0).
- **«Oracle Always Free A10»** — в официальном Always Free GPU shapes **нет**.
- **«GitHub Codespaces GPU»** — снят с продажи (авг 2025).
- **«SageMaker Studio Lab для всех»** — **закрыт для новых** клиентов.

## 2. Что реально тянет обучение U-Net без оплаты

Критерии: PyTorch/CUDA, ≥12–16 ГБ VRAM желательно (или градиентный checkpoint на 8–16 ГБ), сессии часами, **без карты / $0**, ToS = 1 человек×1 аккаунт, команда сама запускает.

| Приоритет | Бэкенд | Почему |
|---|---|---|
| **A (ядро)** | **Kaggle × 5 участников** | ~30 ч/нед×5 ≈ **150 GPU-ч/нед** на P100/T4×2; API `kernels push`; телефон, без карты |
| **A** | **Colab free** (каждый сам) | короткие прогоны/отладка; нестабилен |
| **A** | **Локальный GTX 1650** | препроцессинг, smoke, мелкий batch, профилирование кода; **не** финальный 30 мин @4090 |
| **B** | **Paperspace Free M4000** | free notebook GPU; capacity/очередь; 6 ч autostop; проверить no-card |
| **B (РФ/CN)** | **Baidu AI Studio** | дневные точки на V100; удобнее с **Paddle**; PyTorch может быть ограничен `[проверить]` |
| **B (CN)** | **Tencent Cloud Studio** | ~1.7 ч T4/день после чекина — мало для fold, ок для smoke |
| **B (CN)** | **ModelScope / OpenI** | по акциям/заявкам; AMD ROCm ≠ CUDA drop-in |
| **C (не обучение)** | HF ZeroGPU | минуты, Spaces/демо |
| **Исключить при no-card** | Modal (карта обязательна), Lightning +25 (карта), AutoDL rental, immers/Cloud4Y PAYG, hyperscaler GPU trial | нарушают политику $0 / no-card |
| **Исключить** | Codespaces GPU, Oracle GPU free, Studio Lab (новые) | недоступны |

**Итого для U-Net k-fold без оплаты:** основной пул = **5×Kaggle + Colab + 1650**; опционально Paperspace Free и CN-лабы, если участник уже в экосистеме и ToS ок. Финальный прогон «30 мин на 4090» **бесплатно воспроизвести 1:1 нельзя** — см. FARM_DESIGN.md / PLAN.


---

<a id="farm-design"></a>

# Дизайн пула (FARM) — КОЗ №3 «Комплексирование»

> Дата: **2026-10-15**. Владелец мониторинга — «Слои»; код диспетчера — Основной.  
> **Важно:** это пул **согласных реальных участников**, каждый из которых владеет своим аккаунтом и **сам** запускает jobs (или через свой отзывный API-токен, **только где ToS провайдера это допускает**).  
> Здесь **нет** раздела «anti-ban» / обхода детекции: никаких трюков с IP, fingerprint, rate-limit gaming, запуска «от имени родственника». Вместо этого — **compliance** (§6).

## 0. Что сознательно исключено
- Мультиаккаунт одного человека, шаринг логинов, «ферма» на родственниках, которые **не участвуют** в работе команды → **высокий риск ToS** (Kaggle прямо: один аккаунт на человека; шаринг пароля запрещён). Родственники **вне базового плана**.
- Платный compute (immers, Cloud4Y, PAYG сверх free credits с картой) — политика команды: **$0 / без карты**.
- Обход географических/санкционных блоков ради «невидимости» — если нужен доступ, участник решает легально сам; оркестратор этого не проектирует.

## 1. Целевая ёмкость GPU-часов (до финала 17–19.11.2026)

Горизонт: **с 15.10 по 19.11** ≈ **5 полных недель** Kaggle-сброса (сб→сб).

### Базовый сценарий (только команда из 5)
| Параметр | Значение | Допущение |
|---|---|---|
| Участников с Kaggle | 5 | каждый верифицирован, 1 акк |
| Квота Kaggle | **30 ч GPU/нед**/чел | floating может быть выше — не закладываем |
| Доля планирования | **80%** | `max_quota_fraction` |
| Полезные часы/нед | 5 × 30 × 0.8 = **120 ч/нед** | |
| За 5 недель | **~600 GPU-ч** (P100 или T4×2) | минус простои/очереди Kaggle |
| Colab | **+0…N** | нестабилен; в план не кладём число |
| GTX 1650 | препроцесс/smoke | не считаем как «T4-эквивалент» |
| Paperspace Free M4000 | опционально | capacity часто 0 в пике |

**Реалистично на обучение U-Net:** ~**500–600 полезных GPU-ч** на Kaggle за период, если дисциплина чекпойнтов и стоп idle.

### Верхняя граница (optimistic, всё ещё легально)
| Добавка | Оценка | Условие |
|---|---|---|
| Kaggle floating >30 | +10–20% | только если UI показал |
| Paperspace Free ×5 | + десятки ч | если capacity есть; 6 ч cap |
| Baidu / ModelScope / OpenI | ad-hoc | если участник в экосистеме CN; ToS ок |
| Research credits (edu) | крупно, но **не в срок** | заявка weeks; не базовый план |
| **Потолок «на бумаге»** | **≲ 800–1000 GPU-ч** | без нарушения ToS и без оплаты |

Всё, что выше — только через **новых согласных участников команды** (не родственников-«пустышек») или выигранные гранты без карты.

## 2. Шаблон реестра слотов

Файл: `pool.yaml` (секреты не сюда). Плейсхолдеры, **без PII**:

```yaml
slots:
  - slot_id: S1
    owner_id: member_01          # псевдоним в репо
    owner_contact_ref: "tg:@member01_placeholder"  # не телефон/ФИО в git
    provider: kaggle
    gpu: "P100|T4x2"
    weekly_quota_h: 30
    used_h: null                 # из UI / kaggle quota
    remaining_h: null
    resets_at: "Saturday 00:00 UTC"
    data_allowed: [public_only, contest_approved_foreign]  # орг. разрешили зарубежье
    status: unknown              # ok|degraded|exhausted|token_dead
    launch_mode: owner_push      # owner_push | owner_token_ci (если ToS ок)
    secret_ref: "1password:KOZ3/member_01/kaggle"  # не значение
    health:
      last_check: null
      last_ok: null
      notes: ""
```

Пять слотов S1–S5 минимум (Kaggle). Опционально S1c… Colab, S1p Paperspace — тот же `owner_id`.

## 3. Онбординг участника (10 минут)

1. Прочитать RISKS_TOS + этот файл (2 мин).  
2. Подтвердить: «я сам владею аккаунтом и буду сам запускать / отозваемый токен только мой» (1 мин).  
3. Kaggle: логин → Settings → Phone verify → создать API token → положить в **свой** 1Password/vault item `KOZ3/<pseudonym>/kaggle` (3 мин).  
4. В UI Notebooks записать: weekly GPU total/used/reset → сообщить «Слои» цифры **без токена** (2 мин).  
5. Fork/clone репо ноутбука обучения; пробный `kaggle kernels push` **своим** токеном на dry-run (2 мин).  
6. Colab (опционально): открыть тот же notebook Copy&Edit, убедиться что GPU доступен сегодня.  
7. Готово: слот в `pool.yaml` = `status: ok`.

Никаких общих `.txt` с паролями, никаких «зайди с аккаунта брата».

## 4. Очередь и диспетчер

Приоритеты (выше = раньше):
1. **train fold** (основное обучение U-Net / k-fold)  
2. **ablation** (heatmap, стекинг, гиперпараметры)  
3. **infer dry-run** (проверка пайплайна, метрики на синтетике)

Поток:
```
QUEUED → SCHEDULED → RUNNING → SUCCEEDED
                 ↘ PREEMPTED → QUEUED (другой слот того же data_class)
                 ↘ FAILED → retry|dead
```

Failover (free-only): свободный Kaggle-слот другого участника → Colab (ручной owner) → Paperspace Free → **local GTX 1650** → (стоп; платного нет).

Чекпойнты: каждые N мин → **S3 Cloud.ru free** (или HF private dataset, если ToS/данные ок) → `resume_from` в следующем запуске. Job id = `(git_sha, config, seed, fold)`.

Диспетчер **не** хранит чужие токены: либо CI с секретом **только владельца слота**, либо владелец получает brief «запусти kernel X» и пушит сам.

## 5. Изоляция секретов
- На владельца: отдельный item в **1Password / Bitwarden / vault** (`KOZ3/<owner_id>/<provider>`).  
- В git: только `secret_ref`.  
- Запрещено: общий чат, общая Google Sheet с токенами, один `secrets.txt` на команду.  
- Отзыв: владелец ротирует token при уходе из слота; «Слои» помечает `token_dead`.

## 6. Compliance (вместо «anti-ban»)
| Правило | Практика |
|---|---|
| 1 человек = 1 аккаунт | сверка owner_id ↔ реальный участник команды |
| Соблюдать ToS и rate limits | не долбить API; не параллелить сверх квоты UI |
| Честный idle | стопать interactive sessions; не жечь GPU впустую |
| Данные | зарубежье — **разрешено оргами**; всё равно минимизировать ПДн; 152-ФЗ для персональных данных команды |
| Родственники-«пустышки» | **вне плана** |
| Обход банов/детекции | **не проектируем** |

## 7. Соответствие ТЗ финала (Docker ≤2 ГБ, 30 мин на 4090)

| Требование | Как без оплаты | Что невозможно бесплатно |
|---|---|---|
| Образ Docker **≤2 ГБ** | multi-stage, slim CUDA runtime, веса **не** в образе (скачать на старте), onnx/torchscript по необходимости | — |
| Прогон валидации **≤30 мин на RTX 4090 24 ГБ** | **эмуляция:** замер на Kaggle T4×2/P100 + локальный 1650 с малым batch → экстраполяция FLOPs/throughput; держать headroom ×2–3 | **гарантированно** уложиться в 30 мин именно на 4090 **без** доступа к 4090 |
| Архив ≤2 ГБ | те же веса + скрипт; без лишних датасетов | — |
| Финальный iron = 4090 | в ТЗ организаторов; у команды **нет** бесплатного 4090 | аренда 4090 / чей-то грант с 4090 — вне текущей $0-политики |

Явно зафиксировать для жюри/отчёта: финальный тайминг на 4090 — **риск**, закрывается либо доступом к 4090 у партнёра/вуза **без оплаты команды**, либо допущением орга о профайле на T4 с коэффициентом (если согласуют).

## 8. Матрица рисков

| Риск | ToS | Правила конкурса | ПДн / 152-ФЗ | Митигация |
|---|---|---|---|---|
| Мультиаккаунт / шаринг | бан Kaggle/Colab | возможная дисквалификация | — | 1:1, без родственников-пустышек |
| Утечка токенов | бан | — | доступ к чужим данным | 1Password per-owner |
| ПДн в ноутбуках на зарубежье | — | орг. разрешили платформы; всё равно минимизировать | согласие команды, без лишних ПДн в датасете | публичные/конкурсные данные по ТЗ |
| Не уложиться в 30 мин @4090 | — | провал метрики времени | — | профилирование заранее; headroom |
| Исчерпание free квот перед финалом | — | нет обучения | — | 80% cap; резерв последней недели |
| Colab/Paperspace capacity 0 | — | срыв сроков | — | Kaggle как primary |
| 152-ФЗ (контакты участников в репо) | — | — | штрафы/жалобы | псевдонимы; контакты вне git |

## 9. Стыковка с ботами
- **Основной** — Docker ≤2 ГБ, resume, `dispatch` prototype, PR.  
- **«Слои»** — брифы, сверка квот, health, правки md; **не** создаёт аккаунты и **не** шлёт оргам.


---

<a id="account-pool"></a>

# Пул аккаунтов (легальный, free-only)

## Правила
- **1 человек = 1 аккаунт** на провайдера, сам регистрирует, сам запускает jobs (или свой отзывный token, если ToS позволяет).
- **Не включать** родственников/знакомых, которые не работают в команде («пустышки») — высокий риск ToS.
- Ротация = между **участниками команды** и free-провайдерами. Платный immers/Cloud4Y — **вне скоупа**.
- Секреты: только **per-owner** 1Password/vault; в репо — `secret_ref`.
- Failover: Kaggle (другие слоты) → Colab (ручной) → Paperspace Free → **GTX 1650** → стоп.

Организаторы разрешили зарубежные платформы (15.10.2026).

## Реестр (`pool.yaml` / `pool.example.yaml`)

См. полный пример в `pool.example.yaml` и слоты в `FARM_DESIGN.md` §2.

Кратко:
```yaml
policy:
  max_quota_fraction: 0.80
  paid_backends: []                 # явно пусто
  failover_order_public:
    - kaggle
    - colab_manual
    - paperspace_free
    - local_gtx1650
  # modal / lightning / immers — не в списке (карта / PAYG)
```

## Очередь
Приоритеты: **train fold > ablation > infer dry-run** (FARM_DESIGN §4).  
PREEMPTED → чекпойнт S3 (Cloud.ru free) → другой free-слот.

## Health-check (ежедневно, «Слои»)
| Проверка | Порог |
|---|---|
| Token read-only ok | 401 → `token_dead`, алерт владельцу |
| Kaggle remaining | &lt; 5 ч → алерт; не планировать &gt;80% |
| Colab GPU сегодня | 2 дня без GPU → `degraded` |
| Paperspace capacity | Unavailable → skip |
| Checkpoint age у RUNNING | &gt; 2× interval → hung candidate |
| **Не** проверяем: балансы immers/карт | — |

## Регистрация (без покупок)
1. Участник сам: Kaggle (+телефон), опционально Colab / Paperspace Free.  
2. Токен → свой vault.  
3. Цифры квоты из UI → «Слои» вписывает в pool.yaml.  
4. Аккаунты «Слои»/Основной **не создают**.


---

<a id="risks-tos"></a>

# Риски ToS и правовые (free-only)

Сверка: **2026-10-15**. Зарубежные платформы **разрешены** организаторами. Paid GPU вне политики.

| Риск | Суть | Последствие | Мера |
|---|---|---|---|
| Мультиаккаунт одного лица | Kaggle ToS: один активный аккаунт; Community Guidelines — бан за duplicates | бан, дисквалификация Kaggle | 1 человек = 1 аккаунт. [ToS](https://www.kaggle.com/terms) |
| Родственники-«пустышки» | аккаунт формально «чей-то», jobs запускает другой | тот же ToS-риск + шаринг пароля запрещён | **вне базового плана** (FARM_DESIGN) |
| Шаринг логинов/токенов | «will not share your account or password» | бан | per-owner 1Password; отзыв токена |
| Обход верификации / CAPTCHA / fingerprint | нарушение ToS | бан, юр. риск | запрещено; compliance, не anti-ban |
| Modal / Lightning «free» с картой | Modal: payment method обязателен; Lightning +25 за карту | неожиданный charge / против политики $0 | **не в пуле**, пока нет гарантии $0 без карты |
| Cloud.ru GPU на бонусах | GPU ВМ **нельзя** бонусами; нужен cash ≥15k ₽ | ложный «бесплатный GPU» | только CPU/S3 free. [docs](https://cloud.ru/docs/virtual-machines/ug/topics/concepts__gpu) |
| VK грант vs GPU | стартап-грант **исключает GPU** | нет free GPU | не планировать VK GPU без проверки ЛК |
| Не уложиться в 30 мин @4090 | нет free 4090 | провал ТЗ по времени | профиль на T4/P100/1650 + headroom; искать $0 4090 у вуза/партнёра |
| ПДн команды в git (152-ФЗ) | телефоны/ФИО в pool.yaml | претензии | псевдонимы; контакты вне репо |
| Нестабильность Colab/Paperspace | нет гарантии GPU | срыв сроков | Kaggle primary; чекпойнты |
| Hyperscaler «free GPU» миф | GCP trial нельзя Add GPUs; Oracle/Codespaces GPU нет | потеря времени | см. OSINT |

**Вывод:** легальный объём ≈ **5×~30 ч Kaggle/нед** (+ Colab/Paperspace Free по остатку) + GTX 1650. Финальный 4090@30мин — **дыры без оплаты**; фиксируем в отчёте.


---

<a id="next-actions"></a>

# Шаги на неделю 15–22.10.2026 (без покупок)

## Роли
| Кто | Делает | Не делает |
|---|---|---|
| **Основной** | Docker ≤2 ГБ, resume, PR; прототип dispatch по команде | не мониторит квоты ежедневно; не создаёт чужие аккаунты |
| **«Слои»** | OSINT/FARM брифы, health, md | аккаунты, код/PR, письма оргам, покупки |

Зарубежье разрешено оргами. **Не покупаем** GPU-часы.

## Собрать pool.yaml команды из 5 (чеклист без трат)
1. Список 5 **реальных** участников (псевдонимы в git). Родственников-пустышек не звать.
2. Каждому — онбординг 10 мин из FARM_DESIGN §3 (свой Kaggle + телефон + свой vault item).
3. Каждый присылает «Слои» **только**: weekly total / used / reset из UI (скрин или цифры). Токены не присылать.
4. «Слои» заполняет `pool.yaml` из `pool.example.yaml` (5× kaggle + опционально colab/paperspace).
5. Smoke: каждый сам `kaggle kernels push` dry-run ноутбука из репо.
6. Cloud.ru: только free S3/CPU под чекпойнты (без GPU cash).
7. Основной: Docker slim ≤2 ГБ; training + resume; оценка throughput на Kaggle vs цель 30 мин@4090.
8. «Слои» ежедневно: health-таблица → сводка Основному.
9. Отдельно решить дыру **4090@30мин**: вуз/партнёр $0 или письменная экстраполяция — без immers.

## Не делать
- Карты в Modal/Lightning «на всякий случай».
- Аренда immers/Cloud4Y.
- Общие файлы с паролями.
- Anti-ban / IP / fingerprint схемы.


---

## pool.example.yaml

```yaml
# Плейсхолдеры only. Секреты не коммитить. Free-only + local 1650.
version: 1
updated: "2026-10-15"
policy:
  max_quota_fraction: 0.80
  preempt_grace_min: 15
  checkpoint_interval_min: 20
  paid_backends: []
  failover_order:
    - kaggle
    - colab_manual
    - paperspace_free
    - local_gtx1650
members:
  - id: member_01
    contact_ref: "tg:@member01_placeholder"
    providers:
      - provider: kaggle
        status: unknown
        gpu: "P100|T4x2"
        weekly_quota_h: 30
        used_h: null
        remaining_h: null
        resets_at: "Saturday 00:00 UTC"
        data_allowed: [contest_approved_foreign]
        launch_mode: owner_push
        secret_ref: "1password:KOZ3/member_01/kaggle"
      - provider: colab_manual
        status: unknown
        gpu: "T4?"
        weekly_quota_h: null
        data_allowed: [contest_approved_foreign]
        secret_ref: null
  # member_02 … member_05 — та же схема
shared_backends:
  - id: local_gtx1650
    owner: lab
    gpu: "GTX 1650 4GB"
    vram_gb: 4
    data_allowed: [contest_approved_foreign]
  - id: paperspace_free
    provider: paperspace
    gpu: "M4000 free"
    note: "6h autostop; capacity not guaranteed; verify no-card"
    data_allowed: [contest_approved_foreign]
    secret_ref: "1password:KOZ3/member_XX/paperspace"
  - id: cloudru_s3
    provider: cloud.ru
    gpu: none
    note: "checkpoints/tiles only; no GPU on bonuses"
    secret_ref: "1password:KOZ3/team/cloudru_s3"
```
