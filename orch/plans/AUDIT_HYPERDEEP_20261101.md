# AUDIT_HYPERDEEP 01.11 (MSK): весь путь до 0.41238 и после

Автор: executor коорда A по t218u. Секретов нет. Все числа взяты из файлов по путям, указанным ниже.
Гейт везде honest-10 (gate_v6c machinery: семейные фолды k=10, пороги пересвипываются в каждом фолде, paired bootstrap по 103 семьям).
Триада приёма: Δ≥0.003, P(лучше)≥0.9, F1 повреждённых курганов не хуже.

## 0. Опорные факты
- LB **0.41238** — архив `submission_bot7cand_v2_eb268d6`. Официальный лог (orch/OFFICIAL_SCORES.md):
  - soft recall 0.622, FP ratio 1.14
  - P/R по модальностям: лидар .516/.504, аэро .091/.013, спутник .456/.255, орто .368/.140, гео .769/.656
  - recall по размерам: .432/.323/.326/.405/**0 (5–10k м²)**/.294
- 30.10 LB упал до 0.37838 (v3a_min). Весь спад пришёлся на геофизику: R .656→.311, потому что DOC-1 убрал PIXGBM. Остальные 4 модальности совпали до 3 знаков (orch/LB_DROP_20261030.md).
- База OOF A (stack_arch) = 0.314615. Archive-like база **B′** (курганы — микродиск 0.5 м в центроиде, точное официальное правило cov/rp) = **0.315800**. Это ровно совпадает с W0 g2_decomp, то есть реконструкция воспроизводима (`out/audit-next/b0x_kur_cen_pairs_exact.py`, `kur_cen_pairs_exact_R0.5.pkl`).

## 1. Находки (severity · evidence · можно ли сейчас на CPU · kill)

| # | Sev | Находка | Evidence | CPU сейчас? | Kill |
|---|---|---|---|---|---|
| F1 | **HIGH (зацепка)** | **FR1h + M_rp(gor+fort) аддитивны.** Против B′: +0.00863, P 1.0, p5 +0.00449; F1 повреждённых +0.0108; new34 +0.0027. Против A: +0.0098, P 1.0. Триада PASS к обеим базам. Сумма отдельных Δ равна совместной (0.008622 = 0.008622). FR1h отдельно: +0.00409 P0.999. rp_gor_fort отдельно: +0.00453 P0.9885. rp на arch даёт ≈0 (−0.0001), на object_poly — −0.00016. | `out/audit-next/b_combo_fr1h_mrp.json`, `logs/b_combo.log` | да (собрать дерево, гео-побайт) | если гео-члены не побайтно равны архиву или smoke показывает изменение числа гео-фич — STOP |
| F2 | **HIGH (шов)** | **На LB эффект M_rp не виден.** В v3a_min M_rp был включён (`radius_m` 0.05 для gor/arch/obj/fort, `anchor="rp"`, проверено в `w4geo/kaggle/ds-v3a-min-check/v3a_min.zip.bin` → configs/default.toml:426-430), но P/R лидара, аэро, спутника и орто совпали с v2 до 3 знаков. Значит, OOF +0.0045 может дать на LB ≈0 (на тесте мало городищ и фортификаций или иная агрегация). В LB_DROP §4 есть только намёк: recall ≤100 м² .432→.450. | orch/LB_DROP_20261030.md:11-14,27; v3a_min default.toml | да (документ) | — (это предупреждение для ожиданий: не обещать +0.0086 на LB, реалистично +0.004 от FR1h) |
| F3 | **HIGH (шов)** | **Crop verify в пайплайне почти не применяется к SpOr-курганам.** `[verify] groups=["sp_focal","sp_focal2"]`, но у первичных SpOr-курганов пайплайна нет threshold_group, и они откатываются к markup `sp_or`. В итоге verify (+0.00414 P0.984, 28.10, мерили на группах таблицы) действует только на sp_focal2. На bb_spor 185_Изборск лог «проверка кропа: смешано 6 кругов» против ~26 сырых. | `out/fr1h-pack/tree/configs/default.toml [verify]`, `expds/pipeline.py:787`, `out/fr1h-pack/bb_spor/*` логи | нет: офлайн-гейт требует фолдовых sp_crop-моделей (board json только на ПК) | LB-проба «groups += sp_or» хуже 0.41238 → откат |
| F4 | **HIGH (шов, корень 0.378)** | **Домен гейта ≠ архив.** В OOF A нет PIXGBM, twins, verify, rerank_arch, а курганы там 4-м кругами, а не микро 0.5. Из-за этого ORG3 и M_rp мерили против A, и 30.10 случился провал. Теперь закрыто частично: база B′ воспроизводит микро-курганы. Twins, verify и PIXGBM по-прежнему не моделируются. | orch/LB_DROP_20261030.md:21; `b_combo_fr1h_mrp_APPROX_R0.5.json` (приближённое правило давало B′=0.3075, точное — 0.3158: база чувствительна к правилу) | да: гейтовать впредь против B′ | любая Δ, измеренная только против A, не основание для LB |
| F5 | **HIGH/решение** | **EN-линия PIXGBM** (geo_pixgbm.py + models/pixgbm) лежит в архиве 0.41238 и в FR1h-zip побайтно, а DOC-1 её запрещает. Допуск к final решает Елена. | `out/audit-next/a_fr1h_integrity.json` (geo-члены identical) | нет | — |
| F6 | MED | **Twins (sp_focal_twin) создаются после порогов**, frame_prior их не трогает. В gate-стеке строк twins нет. Эффект FR1h на twins не измерен. | `out/fr1h-pack/tree/expds/watchdog.py finish()` (порядок: rerank_arch → frame_prior → thresholds → merge → multipole → finds → twins → micro) | частично | — |
| F7 | MED | **Паритет scene_drel FR1h.** Гейт брал bot-10 frame_boxes.json, пайплайн — map_footprint(region_crs). Проверен только smoke на 4 участках. | `out/fr1h-gate/xform_stack_fr1h.py`, `out/fr1h-pack/FR1H_PACK_20261031.md` | да (w5data/sat_raw, 210 участков) | доля участков с |Δdrel|>0.02 ≥5% → перемерить гейт с footprint пайплайна |
| F8 | MED | **Нет config.py метрики**, используются встроенные дефолты compute_metrics (cov=inter/min area ≥0.5 и rp∈G; для точек r_tol). | wt/w5/materials/compute_metrics.py:364-405,909-944 | — | — |
| F9 | MED (процесс) | **DINOv3 использовали** в dino-gor и peak-rerank вопреки запрету PLAN_TO_1111. Оба STOP, в zip не попали. | orch/PLAN_TO_1111_20261031.md, COORD_LOG | — | — |
| F10 | MED | **Sat cheap-union / 8 м кластер — NO-GO.** Кросс-групповой NMS sp_focal×sp_focal2×sp_or на 4/8/12 м: лучший allsp_D4 +0.0004 P0.76, все варианты с FR1h отрицательны. Дубли реальные (D8: 7184 пар, 588 обе попали в один GT), но метрика с порогами их уже поглощает. | `out/audit-next/c_sat_cluster.json`, `logs/c_sat_cluster.log` | выполнено | закрыт без новых данных |
| F11 | MED (метрика) | **Слепые зоны recall:** размер 5–10k м² = 0; аэро R .013, орто .140, спутник .255 (у соперников ~.64). | OFFICIAL_SCORES.md | частично (инвентаризация) | — |
| F12 | LOW | Утверждение t217 «запущен cheap union + 8 м + TTA» не подтверждено: out-каталога не было. Сейчас C сделан (F10), TTA BLOCKED (нужен GPU). | ls out/ | — | — |
| F13 | LOW (гигиена) | Висят локальные kaggle CLI: pid 260922 (status v3amin-chk, с 31.10 18:49 MSK) и pid 675843 (kernels list, с 09:14 MSK). Есть QUEUED TPU smoke без lease: elenakonkova/koz3-acc2-tpu-smoke, dmitrijkv/koz3-tpu-smoke. | `out/audit-next/d_stale_kernels.json` | только документ | — |
| F14 | LOW (гигиена) | check_conflicts ERROR=5: в repo-клоне 81 изменённый tracked-файл, worktree вне wt (expds-koz3, orch/bot-W1/e2e_v7/tree), 2 detached HEAD. HEAD repo 70dedf4 устарел против main d8e0e76. configs/default.toml в git clean. | COORD_LOG 01.11 | да (только read) | — |
| F15 | LOW | S3 rotate всё ещё ждёт Елену. | STATE.md | нет | — |

## 2. Сделано в этом аудите (box CPU)
- **(a) Целостность FR1h — OK.** Архив sha 8f3f2671…, FR1h sha fe8d60b9…; по 50 файлов, 46 identical. Изменены ровно 4: configs/default.toml (+12/−4, только 4 sp-порога), expds/pipeline.py (+19), expds/post.py (+60), expds/watchdog.py (+10). Гео-члены побайтно равны. Секретов и данных нет. → `out/audit-next/a_fr1h_integrity.json`
- **(b) ORG3/M_rp × FR1h на OOF A** → F1, F4. ORG3 сюда не входит: его гео-эффект в OOF A не сопоставим с PIXGBM-архивом, а гейт против v2 на geo5 — отдельная задача.
- **(c) Sat cluster** → F10 (NO-GO).
- **(d) Устаревшие kernels и процессы** — только задокументированы → `out/audit-next/d_stale_kernels.json`.

## 3. Что закрыто и не перезапускается без новых данных
r50c5 REJECT, sp_large STOP, rescue_near_kept/ring FAIL, peak-rerank STOP, dino-gor STOP, within-site rank f127db8 STOP, xlayer (−0.011), thr_relax, ORSP, sat cross-group NMS (F10, новое).
