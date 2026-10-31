# REPORT3 — бот 10: «keep» групп без порога (находка бота 7), 28.10.2026

## Вывод (3 строки)
1. **Артефакт честного табло, не выигрыш.** Без порога на обучающих фолдах остаются только `etgr` и `terfl` object_poly (обе есть лишь в семье Гнёздово) и 1 объект `li_ae`; сдача их **уже выдаёт** (пороги 0,096635 и 0,0), так что LB их давно содержит.
2. Менять в `configs/default.toml` **нечего**. Уровень честного балла при keep выше на +0,018622 (0,314615 → 0,333237, honest-10, A); это сдвиг оценки, а не сдачи. Парные Δ вариантов, не трогающих геофизику, при keep и drop **совпадают до 6-го знака**.
3. Последствие: «unified group» в `sota_stack6` (+0,033748) почти целиком тот же артефакт. При слиянии групп etgr/terfl получают порог от maggr, и 723 объекта перестают выбрасываться. Честный прирост multipole берите из «separate groups» (+0,006767).

## Находки (с путями)
- **Что выбрасывается.** `work3/unseen_groups.py` → `work3/unseen_groups.json` (кривые `work/curves`, 10 фолдов семей):

  | группа / класс | без порога | из них TP | семьи |
  |---|---|---|---|
  | `etgr` / object_poly | 403 | 145 | только Гнёздово |
  | `terfl` / object_poly | 319 | 38 | только Гнёздово |
  | `li_ae` / kurgany_povrezhdennye | 1 | 0 | только Верхмарево |

  Итого 723 = `unseen_dropped` из REPORT.md. В отложенном фолде у группы нет другой семьи, по которой можно подобрать порог, отсюда drop.
- **Как честное табло их считает.** `scripts/scoreboard.py:229` вызывает `honest_counts(curves, folds)` с `unseen="drop"` по умолчанию (`expds/metric.py:569–604`). Порог такой группы получается +inf, и объекты выпадают.
- **Как группы попадают в сдачу.** Цепочка: `expds/watchdog.py:finish` → `post.apply_thresholds` (`expds/post.py:163–177`). Порог там равен `thresholds_by_markup[threshold_group или markup_type][класс]`, иначе `thresholds[класс]`, иначе `default_threshold`; объект не выбрасывается никогда. По `configs/default.toml` (тот же в `38f0986` = сданный архив 0,34796, `git show 38f0986:configs/default.toml`):
  - etgr: `[thresholds_by_markup.etgr] object_poly = 0.096635`;
  - terfl: своей секции нет → `[thresholds] object_poly = 0.0`, до этого детектор режет по `[geophys.terfl.object_poly] min_confidence = 0.1426`.

  Внутривыборочные пороги табло (`/workspace/rerank/stack_arch_base.json`, per_class.object_poly.groups) те же:
  - etgr 0,096635: 387 оставлено, TP 144;
  - terfl 0,0: 319 оставлено, TP 38.

  Значит, keep (порог −inf) почти точно повторяет сдачу: в etgr 403 против 387 объектов.
- **Сверка с LB** (`docs/LB_LOG.md`, строки 15–18). Геофизика 0,791092 / 0,258848 одинакова во всех посылках с 24.10 08:35 до 27.10 23:27. Секция `thresholds_by_markup.etgr` появилась в `c616c19` (24.10 08:52 МСК). Детектор и так режет на 0,0958, поэтому геофизика не изменилась. Архивов `dist/submission*.zip` на этой машине нет (они на ПК, `F:\…`). Есть `bot-8/dist/koz3_base_24bd916_e6d2e77c.zip`; его `configs/default.toml` содержит те же строки (`unzip -p … configs/default.toml`).
- **Замер keep и drop** (`work3/keep_vs_drop.py` → `work3/keep_vs_drop.json`, `.log`):

  | раскладка | drop | keep | keep − drop |
  |---|---|---|---|
  | honest-10 | 0,314615 | 0,333237 | +0,018622 |
  | LOFO | 0,315067 | 0,333689 | +0,018622 |
  | 5 случайных раскладок | — | — | 4× +0,018622 и 1× +0,031676 (seed4) |

  - object_poly F1: 0,1305 (TP 78, FP 68) → 0,3168 (TP 261, FP 607). Это совпадает с ботом 7: 0,1322 → 0,3168 на его точной оценке 0,31767 → 0,33613.
  - Бутстреп семей (2000): среднее +0,01261, p5 −0,00002, p95 +0,02199, **P(Δ>0) = 0,6405**. Встроенный `paired_bootstrap_counts`: P 0,643. Это просто вероятность попадания Гнёздово в выборку. Без 17 участков Гнёздово: 0,320624 при drop и 0,320615 при keep.
  - Взвешенно по классам: object_poly +0,01863, povr −0,000008, остальные 0.
- **Парные Δ не зависят от политики.** A−nogor:
  - drop +0,005057, keep +0,005057 (honest-10);
  - бутстреп P 0,9715 в обоих случаях;
  - LOFO и все 5 раскладок совпадают.

  Значит, прежние решения по EXPERIMENTS.md остаются в силе.

## Ожидаемый прирост на LB
**0.** Объекты etgr и terfl уже в сдаче. По правилу приёмки это и не изменение: П2 P = 0,64 < 0,85. Меняется только честный уровень: при keep он ближе к сдаче (+0,0186). Но пороги и параметры детектора EtGr/TerFl подобраны на той же единственной семье (`configs/default.toml`, комментарий над `[geophys.maggr.finds_points]`: «проверка только на тех же данных»). Поэтому keep-уровень по object_poly оптимистичен.

## Патч
- Конфиг: **не нужен**.
- Инструменты: ветка `bot10-unseen-keep` (коммит 4409c5b поверх a839da2) в `/workspace/koz3_shared/bot-10/repo`, diff `/workspace/koz3_shared/bot-10/unseen_keep_tooling.diff`. Добавлены:
  - `--unseen keep|drop` в `scripts/honest_bootstrap.py`;
  - `--honest-unseen keep|drop` в `scripts/scoreboard.py` (по умолчанию drop, поведение прежнее; печатает «оставлено N»).
- Правило: сравнивать варианты только при **одинаковом** `unseen_dropped`. Иначе Δ содержит артефакт, как в sota_stack6 unified.

## Что нужно от ПК/GPU
Ничего обязательного. Для отчётов уровня можно перемерить базу:
`python scripts/scoreboard.py --work <work> --train <train> --honest 10 --honest-unseen keep`.
Перемер sota_stack6 «unified group» против базы с keep (оба keep):
`python scripts/scoreboard.py --work <work> --train <train> --honest 10 --honest-unseen keep <аргументы опыта sota_stack6 из notes/sota_stack6_2026-10-28.json>`.
Ожидание: от +0,033748 останется порядка separate-groups (+0,0068). Это гипотеза, не замер.
