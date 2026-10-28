# PC_RESULTS (ПК DESKTOP, GTX 1650) — бот-9

База A (honest-10) = 0.3146153, флаги = C:\Sloi\work59\base_flags.txt. Код скоринга: основной репо ПК (C:\Sloi\expds-koz3_v0.2.0\expds-koz3); s4_rerank/инструменты bot-10: C:\Sloi\wt_s4 (bot-9 194a691 + bot10-honest-bootstrap a839da2, merge caa7a69).
Веса качаются на ПК прямо kaggle CLI (kernels output --file-pattern), CopyFromBox не нужен.

## Статус этапов
| # | этап | статус | время (МСК) |
|---|---|---|---|
| 1 | s4 rerank r18_imnet: веса fam10 ×10 (kaggle → C:\Sloi\kout\s4) → s4_rerank --device cuda по stack_base_59 → scoreboard --rerank --honest 10 → П1–П5 | ИДЁТ (загрузка весов) | старт 21:39 |
| 1b | s4 rerank r18_scratch (опц.) | ждёт | |
| 2 | s5 OOF: 02_centers_fixed_out.tar.gz (kaggle → C:\Sloi\kout\s5, параллельно, F: не трогает) → stage → nn_oof --resume r18 Unet / r34 FPN → проверка фолдов vs folds_li_tin_59_t128 → scoreboard | ждёт (загрузка стартует после весов s4) | |
| 3a | пакет crop + kur0.5m + rerank_arch + LOO; crop отдельно + honest_bootstrap seeds 5 + MATCH_SCOPE=site | ждёт | |
| 3b | kurgan_lgbm family-fold OOF → --add li_cpu_oof | ждёт | |

## Результаты
(пока нет)

## lidar59
| этап | статус |
|---|---|
| (a) r34-FPN mask 59: веса s1 fold0–3 + s2 fold4 → nn_oof --resume → fuse/diag → scoreboard --honest 10 | ждёт |
| (b) наш скор в их метрике (li_hard2_fused, 4 класса, LAS-регионы, nested 5 family folds; vs 0.5223) | ждёт, если дёшево |
| (c) ConvNeXt | НЕ делаем |
