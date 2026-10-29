# PC_RESULTS (ПК DESKTOP, GTX 1650) — бот-9

База A (honest-10): на коробке 0.3146153, **код ПК даёт 0.3171349** (стабильно во всех прогонах ПК; Δ парные внутри прогона), флаги = C:\Sloi\work59\base_flags.txt. Код скоринга: основной репо ПК (C:\Sloi\expds-koz3_v0.2.0\expds-koz3); s4_rerank/инструменты bot-10: C:\Sloi\wt_s4 (bot-9 194a691 + bot10-honest-bootstrap a839da2, merge caa7a69).
Веса качаются на ПК прямо kaggle CLI (kernels output --file-pattern), CopyFromBox не нужен.

## Статус этапов
| # | этап | статус | время (МСК) |
|---|---|---|---|
| 1 | s4 rerank r18_imnet: веса fam10 ×10 (kaggle → C:\Sloi\kout\s4) → s4_rerank --device cuda по stack_base_59 → scoreboard --rerank --honest 10 → П1–П5 | **ГОТОВО, ПРИНЯТ** (rerank 21:50, SB 21:53, приёмка 22:05) | 21:39–22:05 |
| 1b | s4 rerank r18_scratch (опц.) | ждёт | |
| 2 | s5 OOF (ОТКЛОНЁН, см. ниже): 02_centers_fixed_out.tar.gz (kaggle → C:\Sloi\kout\s5, параллельно, F: не трогает) → stage → nn_oof --resume r18 Unet / r34 FPN → проверка фолдов vs folds_li_tin_59_t128 → scoreboard | веса готовы: tar 888 276 166 B скачан 21:42, sha fold0/r34 fold4 = коробка, разложены в C:\Sloi\s5\oof_cfix_r18|r34\fold{k}; ждёт F: (после s4) | |
| 2b | bot-2 li-decode dump: копия main (коробка 14d419a + patch 0001, git apply --check OK) в C:\Sloi\wt_lidump; веса li_hard2_oof fold0–4 → C:\Sloi\lidump\work; nn_oof --predict-only, EXPDS_LI_DUMP=C:\Sloi\lidump\li_hard2_probs → коробка bot-2/li_dump/ | **ГОТОВО**: LIDUMP EXIT 0 22:38; dump.tgz 1 245 756 317 B sha256 3e275827…7545 сверен на коробке, распакован в /workspace/koz3_shared/bot-2/li_dump/ (li_hard2_probs 54 npz + li_hard2_oof_dump.geojson + lidump.log) | 22:26–23:05 |
| 3a | ~~свой пакет q3~~ СНЯТ 22:44 (заменён пакетом bot-7, п. 3c); был: пакет crop + kur0.5m + rerank_arch + LOO; crop отдельно + honest_bootstrap seeds 5 + MATCH_SCOPE=site | в очереди C:\Sloi\pkg\q3.cmd (после li-dump): база = F:stack_arch, crop = stack_verify_sp_arch, R = rerank_oof --classes arkhitektury, kur = micro_kur (копия bot-7 micro_fam+micro_circles, 0,5 м); кривые 8 вариантов; bootstrap: crop/base, pkg vs base/noCrop/noKur/noR, kur/base, R/base; site — копия wt_s4 + bot-10 site_scope_provisional.diff (C:\Sloi\wt_site) | ETA ~23:00–00:30 |
| 3c | **пакет bot-7** (bot7-candidate 4c3e29e; кит work4/pc, SHA256SUMS ✓ для 3 присутствующих файлов; candidate.diff/candidate.mbox/measure_tools.diff в pc/ отсутствуют) — pc_measure_pkg.ps1 без twins, затем -Twins | идёт (C:\Sloi\expds3c\b7.cmd, лог b7.log); база keep = **0.333237 ✓** | 22:45– |
| 3b | kurgan_lgbm family-fold OOF (`python -m expds.candidates oof --folds 5`, LightGBM по фолдам семей) → --add li_cpu_oof:li_cpu --honest 10 | в очереди q4.cmd (после копии пакета с F:, параллельно CPU-части пакета) | |
| 4 | геофизика bot-3 (GEO_GAP_REPORT §E): 200 и 212/MagGr цвет_2 — geo_predict A2_R3 (maps=all, dedupe=false, pole_circle 0,05), A2, A3 (band auto) на ветке bot3-geophys + cherry-pick c315a9b (C:\Sloi\wt_geo); scoreboard --honest 10 --honest-unseen keep/drop (+ --geo-experiment для Δ) скриптом bot-10 bot10-site-scope-provisional (C:\Sloi\wt_b10) → bot-3/geo_gap/pc/ | в очереди C:\Sloi\geo\geo.cmd — последним (ждёт pix_done.flag после п. 5), выход только в C:\Sloi\geo | |
| 5 | PIXGBM bot-3 (вариант S) + main.py RAM (bot-7) | пороги *_pix подобраны (коробка); тест-кандидаты/хук на ПК — в очереди C:\Sloi\pix\pixmain.ps1 (после lidar59) | |

## Результаты
### s4 rerank r18_imnet (fam10, cuda) — ПРИНЯТ
honest-10 0.317135 → 0.324420, **Δ +0.00729**, 90% [+0.00286; +0.01313], P(Δ>0)=1.000 (bootstrap семей 2000); lb_sim регионы P=0.98.
- П1 ✓ (≥+0.002); П2 ✓ (P≥0.85)
- П3 ✓ LOFO +0.00809; раскладки 5/5 > 0 (+0.0122/+0.0090/+0.0080/+0.0133/+0.0083); honest5 +0.0076, honest20 +0.0075
- П4 ✓ dF1 povr +0.0220, tselye +0.0034, gor 0; new34 +0.0107, optical +0.00003
- П5 ✓ Δ ≥ 0.006 → принимается отдельно (MDE одност. 0.0067)
Файлы ПК: C:\Sloi\work59\stack_s4_imnet.geojson, sb_s4_imnet.json, acc_s4\accept_s4_imnet.json.

## lidar59
| этап | статус |
|---|---|
| (a) r34-FPN mask 59: веса s1 fold0–3 + s2 fold4 → nn_oof --resume → fuse/diag → scoreboard --honest 10 | веса на месте; в очереди q4.cmd после kurgan_lgbm (--encoder resnet34 --arch FPN --resume) |
| (b) наш скор в их метрике (li_hard2_fused, 4 класса, LAS-регионы, nested 5 family folds; vs 0.5223) | ждёт, если дёшево |
| (c) ConvNeXt | НЕ делаем |

### s5 OOF (cfix) — ОТКЛОНЁН (оба)
Фолды 0–4 = configs/folds_li_tin_59_t128.json ✓.
- r18 Unet: Δ −0.0081, 90% [−0.0146; −0.0028], P 0.004 (new34 +0.0143, optical −0.0005) → П1/П2 ✘
- r34 FPN: Δ −0.0030, 90% [−0.0052; −0.0011], P 0.008 (new34 +0.0032) → П1/П2 ✘
Файлы: C:\Sloi\s5\sb_cfix_r18.json, sb_cfix_r34.json.

### 3b kurgan_lgbm family-fold OOF (li_cpu, 59 LAS, 44 семьи, 5 фолдов) — ОТКЛОНЁН
honest-10 0.317135 → 0.316825, **Δ −0.00031**, 90% [−0.0008; −0.0001], P 0.000; 34 новых −0.0028. In-sample +0 (группа li_cpu порогом выключена). Файлы: C:\Sloi\kur\li_cpu_oof.geojson, sb_li_cpu.json.

### 3c пакет bot-7 — отклонения от инструкции (сообщить bot-7)
- Пути C:\Sloi\expds3c\{bot7,in,pkg,wt_bot7} вместо C:\expds3c (разрешение только на C:\Sloi); копия скрипта pc_measure_pkg_sloi.ps1: $Wt → C:\Sloi\expds3c\wt_bot7.
- worktree НЕ через git основного репо ПК (24bd916 там нет): собран на коробке = 24bd916 + git am candidate_measure.mbox (18 коммитов, HEAD 3b407e6), отправлен tar; models — жёсткие ссылки на модели ПК (sha всех .pt = коробке) + rerank_arch_lgbm.{txt,json} из дерева.
- **Баг скрипта:** `--radius-map ''` в PowerShell 5 теряет пустой аргумент → pkg_fam.py падал, curves_pkg пустые (A preds 0). Исправлено на `'--radius-map='` (pkg_fam: пусто → без кругов; курганы 0,5 м уже в fam_crop_kur). Первый прогон снят, пересчёт с 23:00.
- crop = F:\expds3\work\stack_verify_sp_arch.geojson (1 171 949 537 B; crop поверх stack_arch, тот же, что в q3); stack_verify_sp.geojson (27.10, старая база) НЕ брал. База = F:stack_arch, sha256 b412e299…e253 = zcat /workspace/rerank/stack_arch.geojson.gz.
- notes/regions_las_new.txt в дереве не было — скопирован из wt_s4.

### 3c пакет bot-7, БЕЗ twins — ПРИНЯТ (ПК, 23:13 МСК; C:\Sloi\expds3c\pkg\hb_pkg.json, 471 с)
база keep honest-10 **0.333237 ✓** (LOFO 0.333689); crop 0.337158; **A (пакет) 0.353620** (LOFO 0.352778, new34 0.320778).
- A − base: **Δ +0.02038**, бутстреп семей 90% [+0.0046; +0.0251], **P 0.9985**; LOFO +0.0191; сиды/раскладки 5/5 > 0 (+0.0138…+0.0189); lb_sim регионы P 0.936, семьи P 0.930.
- A − crop: Δ +0.01646, 90% [+0.0008; +0.0208], P 0.995; LOFO +0.0167; 5/5.
- по классам (A−base, dF1): povr +0.0150, tselye +0.0041, gor 0, arch 0, fort 0, **object_poly +0.0984**, **finds_points +0.0522**; new34 +0.0244 → П4 ✓.
- П1 ✓ П2 ✓ П3 ✓ (LOFO>0, 5/5) П4 ✓ П5 н/п → **ПРИНЯТ**. Итог пакета с реранком: +0.02038 + ΔR 0.009536 ≈ **+0.0299** honest-10.
- pkg_fam: rows 1 386 709 → 1 386 100; finds|maggr 3732→2928, object_poly|etgr 403→721, maggr 146→0, maggr_poly 0→247, terfl 319→95; stack6 true, twins false.
- **Баг №2 скрипта:** `$tw = if ($Twins) { @('--twins') } else { @() }` — PS разворачивает одноэлементный массив в строку, сплэт `@tw` даёт `- - t w i n s` → pkg_fam падал, прогон -Twins мерил пустой A. Исправлено на `$tw = @(if ($Twins) { '--twins' })`, -Twins перезапущен 23:14.

### 3c пакет bot-7, С twins — ПРИНЯТ (ПК, 23:28 МСК; hb_pkgT.json; прогон после фикса $tw)
pkg_fam: rows → 1 390 172 (+ae_twin 5, +sp_focal_twin 4067 в kurgany_povrezhdennye; остальное как без twins).
**A(twins) 0.354084** (LOFO 0.353262, new34 0.320778).
- A − base: **Δ +0.02085**, 90% [+0.0053; +0.0257], **P 0.9995**; LOFO +0.0196; 5/5 (+0.0142…+0.0200); lb_sim P 0.945/0.945.
- A − crop: Δ +0.01693, 90% [+0.0019; +0.0213], P 1.000; LOFO +0.0171; 5/5; lb_sim семьи P 0.954.
- классы: povr +0.0166 (без twins +0.0150), остальное как без twins → П4 ✓.
- twins сверх пакета: +0.00047 honest-10 (+0.00048 LOFO) — мало для отдельной приёмки (П1/П5), но не вредит; в составе пакета ✓.
- **Итог с реранком: +0.02085 + 0.009536 ≈ +0.0304** (без twins ≈ +0.0299).

### 4 lidar59 r34 (FPN mask, 59 LAS) — ОТКЛОНЁН (ПК, 23:21 МСК)
Фолды = /workspace/oof59_pc/report.json ✓ (тест-семьи и test_regions 6/1/13/18/21 совпадают во всех 5 фолдах; foldchk match=True ×5).
fuse: 1 025 404 объектов, копий 1630, unmatched 0.
- honest-10: 0.317135 → 0.309555, **Δ −0.0076**, 90% [−0.0180; +0.0014], **P 0.081**; 34 новых −0.0367; без лидара +0.00003.
- in-sample: Δ −0.0012, P 0.341.
- П1/П2/П4 ✘ → **ОТКЛОНЁН**. Файлы: C:\Sloi\oof59\r34\li_r34_59_fused.geojson, C:\Sloi\oof59\sb_r34.json, diag_r34.json.

### Бенчмарк PIXGBM bot-3 (ПК, 23:28–23:34 МСК) — C:\Sloi\bench\{bench.jsonl,bench.rss,bench.log}
ПК: Intel Core i5-13600KF (14 ядер: 6P+8E, 20 потоков), 31,8 ГБ RAM; Windows, Python 3.12.10, lightgbm 4.7.0, numpy 2.5.4.
Адаптация: bench_win.py (= bench.py, пути ПК: репо C:\Sloi\wt_pix = bot3-pixgbm 7cdace5, модели C:/Sloi/wt_pix/models/pixgbm/all, данные F:\2\train-expds-3; resource→psutil peak_wset; getloadavg→cpu_percent) + run_bench.ps1 (по одному процессу, OMP_NUM_THREADS, сторож 1,95 ГБ). Те же 7 участков 201/204/208/212/216/217/200, 43 растра, 29,66 Мпкс — совпадает с коробкой. Бенч стартовал после окончания hb_pkgT (23:27:55); параллельно ничего тяжёлого не шло.
| участок | nt=2, с | nt=8, с | пик RSS, МБ |
|---|---|---|---|
| 201 | 5,84 | 2,68 | 274 |
| 204 | 19,10 | 8,57 | 409 |
| 208 | 1,65 | 0,77 | 189 |
| 212 | 46,68 | 21,37 | 447 |
| 216 | 22,85 | 13,98 | 327 |
| 217 | 78,40 | 48,84 | 354 |
| 200 | 59,19 | 30,29 | 792 |
| **сумма** | **233,7** | **126,4** | макс 792 |
Этапы (wall, с) nt=2 / nt=8: load 10,9/3,5 (холодное чтение F: на первом проходе), trend 3,0/3,0, **feat 54,7/54,6 (однопоточный, не масштабируется)**, pred_poly 107,1/42,3, pred_finds 51,9/16,9, cand 6,2/6,2.
**Аппаратный фактор к коробке (8× Xeon):** при 2 потоках 274,5/233,7 = **×1,17** быстрее (на поток); ПК 8 потоков против коробки 1 поток: 420,4/126,4 = ×3,33; против коробки 2 потока: 274,5/126,4 = ×2,17. LightGBM-часть (pred) ПК nt=2 158,9 с против коробки 167,5 с (×1,05); feat 54,7 против 89,4 (×1,63). Экстраполяция bot-3 «все 200–217 nt2 = 646 с» → ПК nt2 ≈ 550 с, nt8 ≈ 300 с.

### 5 PIXGBM на трейне + main.py RAM (ПК, pixmain, старт 23:34:14 МСК)
- PIX_S (geo_predict, pixgbm enabled+replace, 4 порога bot-3): **EXIT 0, 535 с, пик WS 3 052 МБ**. Хук работает: 26 226 объектов, все в pix-группах — maggr_pix finds 19 423 / object_poly 3 928, etgr_pix op 1 012, terfl_pix op 1 863. C:\Sloi\pix\geo_pix_S_train.geojson.
- main.py (wt_bot7, нативно, GPU GTX 1650) на крупнейшем участке 093 (25,5 ГБ, 1 ортофото, LAS 0): **EXIT 0, 1 624 с (27 мин), пик WS 2 507 МБ**, 23:43:09–00:10:12 МСК. Узкое место — чтение F:: ~23–49 МБ/с, к 00:04 прочитано 36 ГБ при файле 25,5 ГБ (несколько проходов по растру), CPU ~30 с за 21 мин, GPU ~1,1 ГБ / 12 %. Для бюджета ≤1200 с: на диске F: ПК один такой участок бюджет превышает; на платформе зависит от скорости диска. Разбивку по фазам не снял: доступ к ПК пропал в 00:15 (лог C:\Sloi\mram\main.log).

### (a) Платформенная симуляция 4c3e29e + twins, весь mock_val 48 участков (ПК, 00:11–00:21 МСК)
pantum/n7q:20261024, --memory 96g, GPU, сеть none, read-only, uid 12345. Zip только для симуляции: C:\Sloi\sim\submission_sim_pkg7tw.zip (sha ad330d33…). Не в dist, не загружался.
- **Контейнер 620,9 с**, конвейер 615,2 с (прошлый прогон 798 с). Фазы: геофизика 3,2 с (raw 1 068), лидар 157,5 с (136 856), оптика 426,2 с (526), канал σ 16,2 с (144 266). raw 282 716, записано 2 413. PIXGBM в кандидате выключен: строк pixgbm в run.log нет.
- writer check EXIT 0: 2 413 объектов, 3,69 МБ (finds 831, object_poly 320, городища 23, фортификации 125, курганы целые 240 / повреждённые 740, архитектура 134). Лишних файлов в выходе нет.
- **Пик RAM контейнера 4,90 GiB** (docker stats ~1 с), **VRAM макс 3 081 MiB** из 4 ГБ, GPU util макс 100 %, в среднем 21 %.
- ПК: i5-13600KF (14 ядер / 20 потоков), 31,8 ГБ RAM, GTX 1650 4 ГБ, Docker 29.8.2. Данные на F:.

### (b) bot-6 211: MagGr maps="all" (ПК, 00:21–00:24 МСК) — ОТКЛОНЁН
geo_predict maps=all → C:\Sloi\geo211\geo_all.geojson. scoreboard с флагами базы (как в q4) --geo-experiment --honest 10.
- 0,365994 → 0,366352, **Δ +0,00036**, 90% [−0,0018; +0,0035], **P 0,572**; 34 новых −0,0037; без лидара +0,0011. accept=false. finds_points F1 0,5818 → 0,5866.
- П1/П2 ✘. Разбивку по участкам не делал (экономия квоты, итог однозначен). kept_base.json / kept_all.json лежат в C:\Sloi\geo211.

### geo.cmd (bot-3 geo_gap): выполнен 00:24–00:31 МСК, выход в C:\Sloi\geo (sb_*_{keep,drop,exp}.json). В bot-3 не копировал.

### Состояние очереди на 00:40 МСК
Сделано: бенч PIXGBM, PIX_S на трейне + main.py 093, платформенная симуляция, 211, geo.cmd. Пороги PIXGBM считались на коробке (раздел 5). Прежняя очередь закрыта.
Идёт: перемер bot-7 v2 (C:\Sloi\expds3c\v2meas).

## v2 перемер (29.10 ~01:30 МСК) — НЕ ГОТОВ
- Прогон в C:\Sloi\expds3c\v2meas, копия pc_measure_v2_bom.ps1 (BOM, Continue). Проверка k10 OK: A_keep 0.333237 / A_drop 0.314615.
- Баг скрипта (строка 44): в curves_S/s6/Ss6 копируются ВСЕ pkl из curves_A, а family_curves.py пропускает существующие (dst.exists) → S=s6=Ss6=A, Δ=0. Числа недействительны.
- Фикс: копировать только pkl неизменённых семей (3 семьи в fam_*). Применение на ПК заблокировано Auto-review (ошибка классификатора) — ждёт ручного одобрения.
- tools_v2.tar на ПК = 39407205… (как в новом SHA256SUMS); expds/ b1a9d28 = af97af0 → перемер без zip один на обе версии.
