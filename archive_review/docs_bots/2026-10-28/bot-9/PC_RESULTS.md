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
