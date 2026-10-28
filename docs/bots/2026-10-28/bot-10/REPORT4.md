# REPORT4 bot-10 — задача 3 (28.10.2026, 21:17 МСК)

## Вывод (3 строки)
1. sota_stack6 **separate** даёт реальный выигрыш: при одинаковом unseen keep +0,0151 (drop +0,0067), site-scope тот же знак; заявленные +0,0337 «unified» — артефакт смешения keep/drop (0,0186). Принять separate условно (P структурный: геофизика в 3 семьях) → LB.
2. rerank_arch проходит строгий П3 (nested, пороги inner-5): +0,0095, P 0,957, LOFO +0,0079, 5/5, site +0,0096; но с поправкой на отбор ≈ +0,0035 и держится на rg_* признаках участка. Micro-circles arch — межучастковый артефакт (site +0,0002) → отклонить; kurgany 0,5 м — слабо (LOFO +0,00006), только в пакете.
3. Пакет без crop (kur 0,5 м + rerank_arch) +0,0109, P 0,979, LOFO +0,0079, 5/5, heavy +0,0014; crop × kur не аддитивны — финальный пакет мерить на ПК.

## Находки (файлы — `/workspace/koz3_shared/bot-10/work4/`)
| что | Δ k10 / P / CI | LOFO / семена | site | файл |
|---|---|---|---|---|
| stack6 sep drop (A 0,314615) | +0,006703 / 0,948 / [0; +0,0197] | +0,006703 / 5/5 | +0,006699 | stack6_global.json, stack6_site.json |
| stack6 sep keep (A keep 0,333237) | +0,015057 / 0,948 | = / 5/5 | +0,014454 | там же |
| stack6 unified keep | +0,015057 (=sep) | | +0,014454 | там же |
| ae_fort_mask | +0,005240 / 0,926 / [−0,00058; +0,00915] | +0,006843 / 5/5 | +0,005260 | aefort_global.json, aefort_site.json |
| rerank_arch strict (R) | +0,009536 / 0,957 / [+0,00063; +0,01535]; без Елизаветинского +0,006792 | +0,007869 / 5/5 (+0,00241…+0,01051) | +0,009590 (P 0,980), site LOFO +0,008152, семена 5/5 | rerank/res_*.json, rerank/*.log |
| R без rg_n/rg_pct | +0,004407 / 0,859 | +0,003748 | +0,004581 | rerank/k10.log, lofo.log |
| R без всех rg_* | −0,003054 / 0,211 | −0,000614 | | там же |
| arch micro (archM) | +0,001076 / 0,867 | +0,001531 / 5/5 | **+0,000227** (P 0,63) | bot7_global.json, rerank/k10_site.log |
| kurgany 0,5 м (kur) | +0,001371 / 0,953 | **+0,000064** / 5/5 | в работе (curves_site_b7kur) | bot7_global.json |
| bot7 bundle | +0,002447 / 0,995 (= bot-7) | +0,001596 / 5/5 | | bot7_global.json |
| R+kur | +0,010907 / 0,979 / [+0,00181; +0,01663] | +0,007933 / 5/5 | | package.json/.log |
| R+kur+archM | +0,011955 / 0,991 | +0,008285 / 5/5 | | package.json/.log |

Leave-one-out пакета (k10): −kur −0,00137, −archM −0,00105, −R −0,00951. Классы аддитивны (пороги по классам), проверка `_check_base_rerank_minus_mine = 0`. Heavy: только kur (+0,001371), new34 +0,000967; прочие 0.
Перекрёстная проверка bot-5 (`bot-5/analysis/joint/summary.json`): мой archM +0,001076 = их M +0,001076; R canon +0,009744 vs мой строгий +0,009536; RM−R у меня +0,00105 (k10), seed1 −0,000015 — согласуется с их «micro поверх rerank ≈0».
Отбор: ~7 вариантов реранкера (all/arch/finds/select/subclass…), σ≈0,0045 → поправка ≈ −0,006 → ≈ +0,0035; Šidák P≈0,74 (совпадает с «select per class» +0,0037, P 0,685 в `/workspace/rerank/REPORT_rerank.md`).

## Ожидаемый выигрыш (честное табло, не LB)
stack6 sep ≈ +0,007…0,015; rerank_arch ≈ +0,0035…0,0095; kur ≈ +0,001; crop +0,004 (не перепроверен). Шум LB SD ≈0,005 (`REPORT.md`).

## Патчи / ветки
- Инструменты: `bot10-honest-bootstrap` a839da2, `bot10-unseen-keep` 4409c5b (в `bot-10/repo`). `bot10-site-scope-provisional` 68710a9 — **заменён** патчем bot-8 (`bot-8/patch/bot8-site-scope-worktree.diff`; сверка `work4/check_site_scope.json`: совпадает).
- Реранкер строгий: `work4/rerank/nested_lofo5.py`, `build_arch_table.py`; пакет: `work4/compose.py`.
- bot-7: ветка `bot7-bundle` 42ada8d (`bot-7/bundle.diff`) — брать только часть kurgany 0,5 м.

## Команды на ПК
См. «Сводка решений» в `/workspace/koz3_shared/EXPERIMENTS.md` (ACCEPTED / REJECTED / PENDING-on-PC с командами `scripts/scoreboard.py --honest 10 ...`). Главное: совместный пакет crop + kur 0,5 м + rerank_arch (+stack6 sep, `--honest-unseen keep` для базы и варианта одинаково) и LOO по компонентам; site-проверка `EXPDS_MATCH_SCOPE=site`.

## Обзоры — `REVIEW.md` (раздел «Задача 3»)
