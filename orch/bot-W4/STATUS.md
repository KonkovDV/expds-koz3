# bot-W4 (гейт, сборки, комплаенс) — время MSK | бот | что | число | ссылка

29.10 22:00 MSK | W4 | прочитаны BRIEF_COMMON/PROTOCOL/LEASES/local_ctx; клон /workspace/koz3/w4/repo (blobless) | — | —
29.10 22:15 MSK | W4 | ветка bots/w4-gate @ 8a11895 (от v3fix 0037c0e): default.toml флаги выкл (sp_dfine, FR1h), configs/v3_on.toml, docs/STATUS.md, docs/LB_LOG.md (+0.41238) | — | github KonkovDV/expds-koz3 bots/w4-gate
29.10 22:15 MSK | W4 | тесты (venv /tmp/w4venv): fr1h+v3_flags+candidate_v2+check_submission 50 passed / 4 failed — те же 4 падают на 0037c0e (нет models/pixgbm/all/poly.txt в *-nomodels) | 50/4 | —
29.10 22:15 MSK | W4 | v3 D-FINE R−A | +0.019802 P 0.976 | bot-10 REPORT10_accept §1
29.10 22:15 MSK | W4 | v3 FR1h TOFR−TO (без R); при R на трейне эффект 0 | +0.004042 / 0 | REPORT11_audit C3h; PREREG_aeor §1
29.10 22:15 MSK | W4 | БЛОКЕР перезамера v3 пакетом: нет table.pkl / OOF-стека A (F:/expds3/work на ПК) на коробке | — | —
29.10 22:15 MSK | W4 | koz3-bot8-oof-honest10 = OOF_MODE=weights (сданные веса на всём трейне, in-sample) → не годится как база гейта против A=0.314615; только калибровка порогов | — | pull/koz3-bot8-oof-honest10 (oof_kaggle_jobs docstring)
29.10 22:30 MSK | coord→W4 | PIXGBM: старый dmitrijkv/koz3-bot6-pixgbm-cleanroom DONE за 8 с ВХОЛОСТУЮ (геофизика 200–217 не входит в xfer-шарды, poly/finds не обучены). Запущен dmitrijkv/koz3-w4-pixgbm-s3 (CPU, internet): тянет 200–217 (~3.7 ГБ) из S3 train-expds-3 → retrain. Код w4pix/k, вывод → w4pix/out2. | аренда acc2/cpu | —
29.10 22:30 MSK | coord→W4 | Данные с ПК (stack A, table.pkl, fold-веса) НЕ скопированы: Shell на ПК = spawn EPERM, Read вне C:\Users\Пользователь отклонён. Блокер у Елены/родителя. Сырьё — только из S3 (local_ctx/s3_train.env, не коммитить); список ключей: /workspace/koz3/s3_keys.tsv | блокер | —
29.10 22:30 MSK | W4 | koz3-w4-pixgbm-s3: v1 ERROR 22:17 (No koz3-xfer-sNNN under /kaggle/input), перепущен координатором 22:18 → RUNNING; W4 не трогает, аренда acc2/cpu у bot-W4-gate до 08:00 | — | w4pix/out1 (лог v1)
29.10 22:30 MSK | W4 | КОМПЛАЕНС: ядро вендорит expds/geo_pixgbm.py (EN-линия) → это P-retrain (воспроизведение моделей), НЕ clean room по PIXGBM_DECISION; P-clean не начат | — | docs/PREREG_w4_pixgbm.md
29.10 22:30 MSK | W4 | de1edc9 флаг выкл: read_band побайтно = v3 (12 растров × 5 режимов, sha 9fcb0f24…), тесты те же падения; части в geo_pixgbm.py (de1edc9, 1327e88) W4 не открывал | OK (частично) | bots/w4-gate 22a32c3 docs/STATUS.md
29.10 22:30 MSK | W4 | геофизика только 200–217: 302 файла, 1,43 ГиБ (s3_keys.tsv); шарды koz3-w4-geo-s000/s001 описаны в PREREG, не созданы (ядро берёт из S3) | — | —
29.10 22:55 MSK | W4 | REPRO v2: main 70dedf4 + 4 release-артефакта eb268d6 ≡ пересборка v2, result.geojson побайтно на synth/210/214/217/044/181; причина −0.012 — нет geo_pixgbm.py + pixgbm poly/finds + rerank_arch_lgbm в main (фолбэк в классику) | OK | bots/w4-repro-v2 56cdf39, docs/REPRO_V2.md
29.10 22:55 MSK | W4 | Остаток: 210 записано 380 (коробка, обе сборки) vs 383 (лог бота-7) — окружение/вход; zip sha8f3f2671 только на ПК | блокер (малый) | bots/w4-gate docs/STATUS.md
29.10 22:55 MSK | W4 | КОМПЛАЕНС: в main tracked копии geo_pixgbm.py в analysis/ (bot-4 review, bot-7 work5/pc/tools/v2) — решить координатору | флаг | —
30.10 00:45 MSK | W4 | GEO-ORG1 (pix_org, org_maggr=false) 31a66bc: C-et+ORG1 против C-et, s0, 17 уч. без 200: both Δ+0.0017 (finds +0.042, poly −0.025), pts Δ+0.0042 P_fam≥0.95 2/5, poly Δ−0.0025 → ORG1 сам по себе НЕ принят (порог +0.009). Пакет C-et+ORG1both против A: Δ+0.0204, P_fam≥0.95, 4/5, срезы OK → прокси-гейт ДА (1 сид, без 200) | FAIL (ORG1) / прокси OK (пакет) | /workspace/koz3/w4geo/eval/R17org_vsCet_s0.json, R17org_vsA_s0.json
30.10 00:45 MSK | W4 | GEO-ORG2 (org2_band_mode="jet", org2_mask для EtGr) за флагами, тесты 6/6; замер требует переобучения pix EtGr (новый кэш --band-mode jet --org-mask) — не начат: коробка 16 ГБ занята fit_all s1/s2 | не замерено | 31a66bc
30.10 00:45 MSK | W4 | docs/ai_declaration.md черновик для Елены (п. 8.5.3/13.6 ТР; шаблона организаторов на коробке нет) | черновик | bots/w4-gate 8f527bb
30.10 00:50 MSK | W4 | пакет C-et+ORG1pts против A: Δ+0.0229 (finds +0.042, poly +0.187), P_fam≥0.95, 4/5, срезы OK, F1 не ниже A → прокси-гейт ДА (s0, 17 уч. без 200); pix_org.load без массива расстояний EDT (память) | прокси OK | eval/R17orgpts_vsA_s0.json
30.10 00:55 MSK | W4 | GEO-ORG3 PREREG записан до замера | — | c5e5c7e docs/PREREG_w4_pix_clean.md
30.10 00:58 MSK | W4 | GEO-ORG3 код 30c1d34+823bda8 (pix_org_et.py, флаг org_etgr=false, compose --orget-tag, 6 тестов OK); OOF geo5 s0 (EtGr только 212–217) | vs A17: A+ORG3 Δ+0.0206 (poly +0.206; F3 +.007 F4 +.017 F5 +.080; P_region 0.9995, P_family 0.70, 3/5) — совпадает с эталоном орг. C_et_poly +0.0208; C-et+ORG3 +0.0225; C-et+ORG1pts+ORG3 +0.0267 P_fam 0.97 4/5 ГЕЙТ ДА; C-et+ORG1both+ORG3 +0.0238 ГЕЙТ ДА | eval/R17orget_vsA_s0.json
30.10 01:05 MSK | W4 | GEO-ORG3 итог s0 vs A17: A+ORG3 +0.0206 (P_fam 0.70 структурно: EtGr только 212–217, 3/5); A+ORG1poly+ORG3 +0.0194 P_fam 0.97 4/5 ГЕЙТ ДА (аналог орг. C_mgpoly_et +0.0206, 5/5); A+ORG1both+ORG3 +0.0236 ДА; C-et+ORG1pts+ORG3 +0.0267 ДА (лучший). Инференс 216: 6.6 с, 153 МБ. Модели full: /workspace/koz3/w4geo/models_orget/org (пороги 0.1/0.1), не в репо | eval/R17orget*_vsA_s0.json
30.10 01:10 MSK | W4 | PREREG модальностного гейта C-et+ORG3 (a/b/c, честно: после прежних замеров; ORG1-точки исключены) записан ДО финальной проверки | — | bots/w4-pix-clean 5702c78
30.10 01:20 MSK | W4 | кэш классики 200: box_mean поблочно (побайтно тот же, тест на 4 формах), пик 4711 МБ (было OOM при 7,5 ГБ), 598 объектов, pkl побайтно = кэш ре-гейта | OK | 8bb5803 expds/accel.py
30.10 01:25 MSK | W4 | ФИНАЛ модальностный гейт 18 уч. C-et+ORG3 vs A18 (score2 0.4227): (a) 12/12 уч. без EtGr побайтно = A; (b) Δ +0.022499, p_region(EtGr 212–217) 1.0; (c) ΔF1 poly +0.225, finds 0 → PASS. A18/C18 geojson побайтно = файлам ре-гейта координатора (sha 02281f62…, a831d66f…). P_family 0.70 справочно | PASS | runs/FINAL18_modality_gate.json, scripts/pix_modality_gate.py
30.10 01:35 MSK | W4 | v2-clean собран: koz3_v2clean_8bb5803_76de89fc_wb042594f.zip 542 МиБ 58 файлов sha256 81b48731d53994ebb3211ed68a96eca3d7e215215daea27931c70bcc1767f968; configs/v2_clean.toml (impl clean, maps etgr+terfl, org_etgr=true, пороги etgr_org 0.21579 / terfl_pix 0.58017 — sweep по OOF 18 уч.); DOC-1 OK (нет geo_pixgbm/geo_finds_gbm/models/pixgbm); check_submission --no-smoke: FAIL только окружение коробки (нет torch); смоук геофизики из архива на 216: 292 etgr_org, 34 с | собран, НЕ отправлен | /workspace/koz3/w4geo/dist/, runs/*manifest.json, runs/v2clean_files.txt
30.10 01:55 MSK | W4 | pytest bots/w4-pix-clean: до 9F/422P/32S/2E → после 0F/435P/38S/0E (8bb5803); bots/w4-doc1-v4 db92ded (от bot5-candidate-v4): 8F/501P/49S/2E → 6F/503P/49S/2E (остаток — нет torch/cv2 на коробке) | OK | —

## 30.10 01:57 MSK — v2-clean пересобран (замечание бота 1)
TerFl pix выкл.; etgr_org=0 (N0, PREREG 5ea8aea, гейт FINAL19_gate_N0 PASS Δ+0.023191; sweep-вариант Δ+0.020604 PASS).
Архив 26ad1ba sha256 15d59e9f7c120b56079f4cc7bc8cb83c14b698db9a57228e75e286dddd9842c9, DOC-1 OK. check_submission: очередь
watcher w4-v2clean-check → usertest1234/koz3-w4-v2clean-check (CPU слоты заняты W3 до ≤06:01), вывод w4geo/kaggle/out_v2clean_check.
Аренды: dataset/kernel usertest1234/koz3-w4-v2clean-check (bot-W4-clean) — снять после.

## 30.10 08:35 MSK — check_submission rerun + independent W1 V7 regate
- check_submission FAIL 13 (koz3-w4-v2cleanchk-k): cause = ENVIRONMENT, not archive. Kaggle image lacks laspy + segmentation_models_pytorch; official image pantum/n7q:20261024 installs them from requirements.lock (laspy==2.7.0, smp==0.5.0) and import-checks them. Archive 15d59e9f is code+weights+.so only (no requirements by design; metadata.json image/entry_point OK). Other checks PASS; WARN unreferenced weights (pix_clean finds/meta/org) is benign.
- Rerun: dataset aaaa0000004/koz3-w4-v2clean-check (ready), kernel aaaa0000004/koz3-w4-v2cleanchk2 (pip laspy[lazrs]==2.7.0 smp==0.5.0, then check_submission), queued as watcher item w4-v2clean-check2 (pull → w4geo/kaggle/out_v2clean_check2).
- Leases: old dataset:usertest1234/koz3-w4-v2clean-check released; old kernel lease already released by watcher 07:55; no bot-W4-clean entries left. New: bot-W4-chk2 (dataset+kernel aaaa0000004, until 20:23 MSK) — to be released after the result.
- W1 V7 independent regate → runs/independent_regate_W1V7.json: Δall +0.00330 (reproduced exactly, table == clusters engine), raw P 0.935 (0.934 @10k boot) < Bonferroni 0.9857 → FAIL; new34 +0.00044, nolidar +0.00873, floors OK; base=joint: Δ +0.0026, P 0.85.

## 30.10 09:40 MSK — check3 PASS, DOC-1 hard guard, push
- check_submission rerun on 15d59e9f (kernel koz3-w4-v2cleanchk2, offline wheels dataset aaaa0000004/koz3-w4-wheels: laspy 2.7.0, lazrs 0.8.2, smp 0.5.0): PASS — FAIL 0, WARN 2 (unreferenced pix_clean weights; unshare -rn n/a), smoke 12.5 s, RSS 1119 MiB, 80 rows. check2 failed only because Kaggle has no internet (pip). Leases bot-W4-chk2 released.
- DOC-1 hard guard: bots/w4-pix-clean 70d335a — scripts/check_archive_clean.py (EN names + scripts/en_denylist.sha256 (35 sha, sha256sum-only) + bytes grep geo_pixgbm|сторонний репозиторий|pixgbm/ (i) + configs/*.toml must not reference w4pix/models/pixgbm/geo_pixgbm/kurgan-pix/hillshade-parity runs); legacy import moved to expds/pix_legacy.py (excluded from archives). tests/test_doc1_guard.py 20 passed incl. e2e negative (planted geo_pixgbm.py → rc≠0, no zip).
- Rebuilt v2clean from 70d335a (same config + 6 weights): guard PASS, sha256 545ef32a70b4994f4d694828ae5eb28f27c9161d7f19a96f5a9d3833b87e093f (changed from 15d59e9f: only configs/default.toml comment + expds/pipeline.py rename; same 58 files). Old 15d59e9f fails the new guard (2 pattern hits). check_submission of 545ef32a queued: watcher w4-v2clean-check4 (lease bot-W4-chk3 dataset+kernel aaaa0000004, until 21:20 MSK).
- Ported to bots/w4-gate ab9fd82 (20 passed). v3 probe build (configs/v3_on.toml) fails the guard as intended: legacy PIXGBM enabled + model_dir models/pixgbm/all + geo_pixgbm in pipeline.py → v3/v4 need a pix_clean port before any archive. Not ported to bots/w4-doc1-v4 (different build script; its default.toml model_dir would trip it).

## 30.10 12:55 MSK — bot C (агент 3) принял роль W4 gate + W5 rerank
- wt/w4 @ bots/w4-gate 7eb271c; wt/w5 @ bots/w5-rerank c4e18a9; orch lease.sh ok.
- Жду гейт после r50c5 f234 (hold до 31.10 03:00). Независимая перепроверка: A=0.314615, Δ≥+0.003, P≥0.9.
- Очередь: v3a / check4(545ef32a) / DOC-1 guard; WS-G1+OOF-реранкер после dino-emb.
- GPU push только с арендой от A. Чужие RUNNING не трогаю. Отчёты A ≤10 строк.
- Блокер: на боксе нет table.pkl/OOF-стека A, out/w5-dino-emb, w4geo/kaggle/out_v2clean_check4 — нужны пути/зеркало от A.

## 30.10 12:57 MSK — зеркала для C
- /workspace/koz3/out/w5-dino-emb восстановлен с Kaggle (emb.npy 327M).
- table.pkl/OOF A: блокер ПК Елены (F:\\expds3\\work → /workspace/koz3/pc_data или w5data/work).
- out_v2clean_check4: нужен pull aaaa0000004/koz3-w4-v2cleanchk2; CPU-пул ключей нет.

30.10 12:57 MSK | bot C | пути A подтверждены: dino-emb OK (emb 327M n~222929); table/OOF A hold (pc_data пуст); check4 пуст — нет токенов aaaa/usertest; GPU не пушу; жду f234→гейт 31.10 03:05
30.10 13:05 MSK | A (coord) | orch refresh: ROSTER.md, STATE NOW, local_ctx/README, GPU cooldown до 31.10 03:05, watcher запущен; аренды D: branch:bots/w1-aeor + branch:bots/w3-kurgan-seeds до 01.11 12:57 | — | orch/ROSTER.md

30.10 13:10 MSK | bot C | GO от A: table/OOF A снят. Старт: паритет v2 → v3a рычаги → WS-G1. A=0.314615 подтверждён. GPU не пушу. check4 ждёт ключи.
30.10 13:11 MSK | C | GO: паритет v2 → v3a → WS-G1/OOF-rerank; box/q1 leased; A=0.314615 d=0

30.10 13:12 MSK | C | Елена via A: цель 0.61 строго. Приоритет: (1) паритет v2 PASS (2) v3a max Δ на аэро/4-й мод. + крупные 5–10k м² (3) OOF-rerank/WS-G1. Ищем прорыв aerial recall (сейчас 0.013); план ~0.45–0.50 недостаточен.

30.10 13:16 MSK | C | BREAKTHROUGH_061 START: parity PARTIAL (нет LAS/tif/v2 zip на боксе; prior golden PASS 56cdf39). L10o/L3a FAIL; L3m Δ+0.008 но d_new34−0.05 FAIL. Далее: #12 → v3a-min(ORG3+M_rp) → xlayer → τ/MagGr → WS-G1 parallel на box/cpu. GPU нет. acc1/cpu-2: готового ядра нет — пока не прошу.

## 30.10 13:24 MSK — bot C: parity / #12 / v3a levers (OOF A, honest-10, PREREG bot-W4/PREREG_v3a_levers.md)
- Parity v2: PARTIAL (pass-with-caveat): 4 v2 release artifacts in git eb268d6 sha256 == REPRO_V2 manifest; golden e2e NOT re-runnable on box (no LAS/tif inputs, no v2 zip, no w4geo/dist). Prior 29.10 byte-identical evidence stands. LB blocker = v2clean 545ef32a check4 (needs aaaa key). runs/parity_v2_20261030.json
- #12 composition gate implemented (public/closed mixes, weighted family bootstrap). runs/compgate_notes.md
- M_rp reproduced on box (rebuilt geom_wkb/gt_wkb + W0 rp pairs): Δ+0.00437 P0.989 new34 +0.0022 opt +0.0085; #12 pub +0.0021 P0.92, closed +0.0038 P0.999 -> PASS = v3a-minimum (ORG3 part not on this OOF). runs/v3a_stack_gate.json
- FAIL: X1 xlayer Li→Ae/Sp, Sp↔Or −0.0110 P0.01 (metric matches globally -> copies can't add hits); L3be break-even +0.0015 P0.87; L3m modality-τ +0.0080 P0.80 but new34 −0.054; L3a area-bin τ −0.0035; L10o obj>30m +0.00001; L14 dedup −0.019; L14s −0.009; L14g no-op.
- Aerial: oracle (any candidate) recall ae kurgan povr 0.45 / tselye 0.45, ae arch 0.0, ae gor 0.37 -> 639 ae povr GT have no candidate: aerial is detector-bound (W1 GPU), not post-proc.
- Blocked: MagGr 0.05 m (inference param, needs raw geophysics), L8 EtGr depth, L13 survey contour (no inputs on box).
30.10 13:24 MSK | C | pushed bots/w4-gate b03a4fd, bots/w5-rerank 8c5c37b; released slot:box/q1 + slot:box/cpu (work done; WS-G1 retry waits ViT-B emb)

30.10 13:26 MSK | C | A ack: ORG3∉LB OK; xlayer закрыт; M_rp PASS. Дальше: (1) check4 ждать aaaa-ключ (2) пакет v2clean+ORG3+M_rp для Елены (3) WS-G1 только после vitb TPU COMPLETE. box/q1→D e2e, box/cpu→B WS-182 — не беру. GPU 03:05 = ae V7/WS-A2.

30.10 13:27 MSK | C | A: usertest+aaaadg keys в local_ctx; aaaa0000004 нет → check4 HOLD. GO pull check4 сразу как aaaa появится.

30.10 13:32 MSK | C | check4 PASS (aaaa koz3-w4-v2cleanchk2, FAIL0 WARN2). PREP_V3A DONE: koz3_v3a_min_4da2eb9_2f2d236a_wb042594f.zip sha256 132c22eb… (v2clean+ORG3+M_rp), DOC-1 OK, README_v3a_min.md. Branch bots/w4-pix-clean @ 4da2eb9. Не сабмитить — Елена. Next: optional check_submission v3a_min on aaaa; WS-G1 after vitb.

30.10 13:37 MSK | C | v3a_min check_submission aaaa/koz3-w4-v3amin-chk: PASS FAIL0 WARN2 (sha132c22eb OK, smoke 14s). Log: w4geo/kaggle/out_v3a_min_chk/check_submission.log. Leases aaaa/cpu-2 + dataset/kernel released. Elena-ready; LB untouched.

30.10 14:27 MSK | C | A PRIORITY: LB 0.412→0.378 = geo PIXGBM cut (DOC-1). v3a_min **NOT next LB**. Path: cleanroom PIXGBM per PIXGBM_SPEC_CLEANROOM + TerFl 0.58017; gate geo P/R ≥ v2 (~.769/.656) vs bot7cand_v2/geo5, not A. Waiting PIXGBM_RECOVER_*.md from coord. No LB submit. WS-G1 hold. Blockers: recover brief not on disk yet; need geo5/v2 baseline artifacts + TerFl re-enable; box still lacks full geo rasters for retrain if needed.

30.10 14:30 MSK | C | Gate prep: geo5 harness FOUND (pix_clean_eval/oof, R17* JSONs, compute_metrics). S3 geo pull started 320f/1.43GiB → w4geo/train. MISSING: bot7cand_v2 zip (Elena PC), oof/geojsons, modality-P/R-vs-v2 wrapper. No aaaa/cpu-2. Await PIXGBM_RECOVER brief for concrete GO.

30.10 14:35 MSK | C | PIXGBM_RECOVER GO: geo pull DONE w4geo/train 1.5G; scripts/pix_geo_pr_gate.py added (C* vs REF, LB geo P/R); v2_clean terfl_pix=0.58017, maps etgr-only until gate. Await Elena v2-ref + A/D OOF finds-fix. v3a frozen.

30.10 14:35 MSK | C | A ack: train/ canonical; v2-ref=Elena; OOF/finds-fix=A/exec, D=CPU folds. C (1)(2) DONE @81d502b pix_geo_pr_gate + terfl_pix=0.58017 maps=[etgr]. Idle for C*/REF geojsons. No aaaa/cpu-2.
## 30.10 15:33 MSK — check4 PASS, new iteration
- check_submission archive 545ef32a (v2clean 70d335a, DOC-1 guard): PASS — FAIL 0, WARN 2, sha_ok=True, wall 48 s (out_v2clean_check4). Lease bot-W4-chk3 released (kernel lease already freed by watcher).
- Lease slot:box/q2 → bot-W4-v3 until 23:32 MSK; branch:bots/w4-gate renewed until 31.10 01:32 MSK.

## 2026-10-30 iteration (bot5-W4)
- check4 545ef32a: PASS (FAIL 0, WARN 2, sha_ok). Lease bot-W4-chk3 released.
- v3_on: build b505656 → koz3_v3on_b505656_6d4764f8_w6dfd2e07.zip (680.5 MiB, sha ebc8e4ba…), DOC-1 guard PASS (no geo_pixgbm, no models/pixgbm). Flags off: v2clean zip sha 545ef32a… byte-identical.
- v3 gate (honest-10, boot 2000, vs v2clean B0 0.313967): V3P Δ +0.0198 P 0.976 new34 +0.0084 nolidar +0.0751 → PASS; V3F (optimistic) Δ +0.0328 P 0.9985.
- D2 (PREREG becf0c1): D2-1 C vs A Δ −0.00065 P 0.0 (object_poly F1 −0.0065) → FAIL; D2-2 direct parity BLOCKED (no v2 legacy geophys output in clean room), proxy |Δ−0.012|=0.0126 → FAIL; D2-3 G4: v2 post.merge_within_markup isolates etgr (static, same in w4-gate); dynamic: isolated is better than merged by +0.00016 (1578→180 rows).
- Kaggle check_submission on v3: NOT run (needs separate W4 lease).

## 30.10 16:20 MSK — origin/bots/w4-gate six "W4 C" commits: NOT OURS → not merged
b03a4fd (13:24, author "W4 gate bot <w4-gate@koz3.local>": v3a levers PREREG/#12 comp-gate/M_rp PASS +0.00437 P 0.989/levers FAIL/parity PARTIAL, 15 files incl. scripts/v3a_levers.py),
f901f89 (13:32 check4 PASS + PREP_V3A), 194fe48 (13:37 v3a_min check_submission PASS), 66c62b4 (14:28 "LB drop ack — v3a_min NOT next LB; wait PIXGBM recover"),
c568499 (14:34 STATUS gate prep + pix_geo_pr_gate), 105ac1b (14:35 align with A) — last five author "bot-W4-gate <bot-w4@koz3.local>".
Identity ≠ bot5-w4, not made in this session, commit messages/COORD_LOG reference PIXGBM non-cleanroom research → in doubt ⇒ no merge.
bots/w4-v3on stays separate (push target for W4 bot5). Contents of these commits were not opened (cleanroom).
# P-W4-compgate + v3_on+M_rp — result (30.10.2026 16:21 MSK), PREREG docs/PREREG_w4_compgate.md (c16be2f, before measurement)
R  rebuild A on box (fresh split+cache): 60/60 cluster blobs equal cached, honest 0.3146153227 → PASS.
Sanity: C0 Σ_mod counts == unweighted (0 mismatched cells, all variants); C1 |w≡1 − honest| ≤ 1.1e-16; M0 extras-without-disk == A and == V3P (0.0).
Weights (target/OOF GT share): PUB geo 1.19 li 1.49 sp_or 0.89 ae 0.13 or 1.03; CLO geo 0.48 li 0.90 sp_or 1.37 ae 1.01 or 11.9 (noisy).
| comparison | UNW Δ (P) | new34 | nolidar | PUB Δ (P) | CLO Δ (P) | comp-OK |
|---|---|---|---|---|---|---|
| V3P vs B0 | +0.019820 (0.976) | +0.008405 | +0.075076 | +0.014806 (0.942) | +0.019615 (0.976) | yes |
| MRP vs A | +0.004670 (0.9925) | +0.002431 | +0.008924 | +0.002353 (0.933) | +0.005075 (0.999) | yes |
| V3P+MRP vs V3P (gate) | +0.002758 (0.9985) | +0.003660 | +0.002369 | +0.001078 (0.998) | +0.002604 (0.998) | yes |
| V3P+MRP vs B0 | +0.022578 (0.9855) | +0.012064 | +0.077446 | +0.015884 (0.948) | +0.022219 (0.986) | yes |
V3P+MRP gate: Δ<0.003 → FAIL by prereg (P, floors, slices, composition all OK); per prereg: M_rp partly overlaps v3_on (+0.00467 on A → +0.00276 on top of v3_on).
MRP on A reproduces W3 +0.004670 exactly. Disclosures: TP attributed to predicted modality; group etgr_org (C19 extras) mapped to geo via the etgr prefix rule (not named in PREREG list).

## 01.11 20:20 MSK — J2 regate PASS, v3onchk PASS (env-pin), Kaggle ownership
- J2 (PREREG 7b5882e, before measurement; rerun with /workspace/koz3/w4venv/bin/python, scipy 1.18.1 — the 30.10 crash was w1/venv without scipy): sanity K=0 mismatches, N reproduces compgate V3P_MRP (|d| 8e-12). J2W4 0.346143 vs V3P 0.333787: Δall +0.012356 P 0.9975 (p5 +0.0041), new34 +0.016405 (P 0.84), nolidar +0.002845 (P 0.9805); floors all ≥0 (kurg_povr +0.031, fort +0.042, object_poly +0.003, finds 0) → PASS (Δ≥0.003, P≥0.9857). α picks nested on v3 [0.6,0.3,0.6,1.0,0.6,0.6,0.3,0.6,0.6,0.6] vs W1-from-A [0.6,0.3,0.6,0.6,0.6,0.3,1.0,1.0,0.6,0.6]. W1's claim +0.0156 shrinks to +0.0124. Secondary vs V3P+M_rp: Δ +0.009598 P 0.9865. → runs/independent_regate_W1J2.json
- v3onchk ERROR 30.10 16:07: run.py v1 used a hard path /kaggle/input/koz3-w4-wheels/laspy… (no recursive glob) → IndexError before the check. v2 (16:33, fixed glob, never logged) COMPLETE but FAIL: no D-FINE reference region 103 (check_submission DFINE_KNOWN, /workspace/data/train-expds-3 absent on Kaggle) + run A vendor regex cp312 cannot load on Kaggle py3.13.
  Fix: own kernel dmitrijkv/koz3-w4-ref103 (S3 via koz3-s3-cred, keys not printed) → SpOr 103 + UTM.json (sha 93e9bd90…) → dataset aaaa0000004/koz3-w4-dfine-ref103; run.py v3 rebuilds region dir and passes --dfine-region.
  v3 result (out_v3on_check3, sha ebc8e4ba OK): B (env transformers 5.21.0 cp313 offline wheels) PASS — FAIL 0, WARN 4, 97.7 s; D-FINE probe 14 queries ≥0.3 on 12 windows (f0 ref 14), max 0.7037 (ref 0.704); smoke sp_dfine n_det 16. A (archive vendor) FAIL 3 = all py3.13-vs-cp312 regex vendor (platform pantum/n7q is py3.12) — environment, not archive.
- elenakonkova/koz3-w4-li-post-gate (COMPLETE 01.11 19:37 MSK): NOT ours — bot-4 li_post line (bots/bot4-li-post b61e3f6 "li_post Kaggle COMPLETE numbers", EXPERIMENTS.csv "via koz3-w4-li-post-gate"); no W4 lease/script. Not pulled. Their recorded result: Li-only honest-LOFO off 0.2803, thr_nested +0.118, thr_nms8 +0.1195.
- elenakonkova/koz3-s3-index (ERROR, 01.11 19:39 MSK): no W4 lease, script or log on box, not W4 prefix → not ours; content not opened, not fixed.
- Leases: box/q1, dmitrijkv/cpu + kernel ref103, aaaa0000004/cpu + kernel v3onchk + dataset ref103 all released.
01.11 20:23 MSK | W4 | v4: аренда slot:box/q1 (bot-W4-v4); RR1s-cf в main (4293458 и др.) целиком под analysis/2026-10-30/bots/bot-W1/** (запрет) → ревью по копии W1 на коробке /workspace/koz3/w1/cf. Ранкер: LightGBM на семью, 50 признаков X4 = X2 (контекст стека, 35) + cnx_* (5, ConvNeXt лидар) + cf_* (10, диски по LAS 0,5 м, кольцо 8 м, земля 2/8); ядро koz3-w5-cloudfeat тянет LAS из S3 (сеть); сохранены только OOF (nested_w5cf.npz), полной модели и модуля инференса НЕТ | модуль не готов → ждём W1 | /workspace/koz3/w1/cf
01.11 20:25 MSK | W4 | PREREG деплоя v4 (α = мода вложенного выбора = 0,6; пороги li-групп по всем фолдам в шкале S, без M_rp) до расчёта; V4C (V3P + RR1s-cf без M_rp, α вложенно) против V3P: Δall +0.009539 P 0.9865 (порог 0.9857), new34 +0.0126, nolidar +0.0005, полы OK. Ориентир parity для v4 ≈ +0.0095 (а не +0.0124 J2: та с M_rp). Опечатка PREREG: «110 регионам» = все регионы трейна | PASS | 8a0d48d, orch/bot-W4/runs/v4_fit.json
01.11 20:27 MSK | W4 | каркас: флаг w5_cloudfeat=false (default.toml), expds/w5_cloudfeat.py (плагин expds.w5cf_infer: region_features()/score(); ref models/w5cf/ref.npz; W5CloudFeatMissing в preflight), хуки pipeline (preflight, лидар-проход, output_floors) и watchdog (finish/safe_finish до apply_thresholds); тесты 11/11, связанные 128 passed 10 skipped; configs/v4_on.toml = v3_on + флаг + [w5cf] + 7 секций порогов li-групп | — | 18ba8b7, 055d3a5
01.11 20:29 MSK | W4 | сборка флаг выкл (055d3a5, configs/v3_on.toml, те же 8 весов+vendor): config sha 6d4764f8 = v3on, 3507 файлов; против манифеста v3on ebc8e4ba: 3504 побайтно равны, отличаются только expds/pipeline.py, expds/watchdog.py (хуки) и новый expds/w5_cloudfeat.py; zip целиком не равен (sha 1c191f39…), т.к. меняется код; поведение при выкл флаге = v3 (тесты identity finish/floors, без I/O). DOC-1: OK. v4_on без модуля падает в начале с понятной W5CloudFeatMissing | parity OK (пофайлово) | /workspace/koz3/w4geo/dist/koz3_v4off_055d3a5_6d4764f8_w6dfd2e07.zip
01.11 20:31 MSK | W4 | шаг 5 (сборка v4_on, smoke, J2 через архив) не выполнен: нет модуля W1. Нужно от W1: модели на всём трейне, инференс X2/cnx/cf из run feats и тестового LAS без сети, parity с OOF, expds/w5cf_infer.py. ref.npz готов: orch/bot-W4/runs/w5cf_ref.npz (1,5 МБ, на коробке). Kaggle check_submission не запускался; аренда снята | ждём W1 | —
01.11 21:25 MSK | W4 | v4 nocnx: cnx_* на инференсе нет (в v3_on только r18/D-FINE/pix_clean/LGBM, li_cnx закомментирован) → вариант nocnx; PREREG 6c2e946 до расчёта. Гейт V4N−V3P Δ+0.011653 P 0.998 (p5 +0.0039), new34 +0.0037, nolidar +0.0005; полы: fortifikatsii −0.00556 < −0.005 → FAIL (runs/independent_regate_W1J2_nocnx.json). Через код архива (OOF-скоры, ref.npz, α 0.6): смесь = fit (maxabs 4e-5), honest-10 Δ+0.0052 P 0.921, fortifikatsii −0.0084 (runs/v4on_archive_gate.json). | FAIL, Kaggle не запускался | runs/*.json
01.11 21:25 MSK | W4 | v4_on собран: koz3_v4on_da537d3_aafe99d4_w644c0017.zip 684,1 МиБ 3519 файлов sha256 34819adbcd070b9894fb9fbc069ef297202bc8f21dda9d1b315158f3733693d4, DOC-1 OK; ранжировщики пересобраны train_final.py --nocnx, sha = W1 w5cf_infer_nocnx (все 8 файлов), в архиве WEIGHTS_MANIFEST.json, preflight сверяет sha. Сборщик: .npz разрешён только для ref.npz и gq_ref.npz с проверкой «только 1-D» (решение по политике — на бота 1). Parity W1 на коде архива: все 5 участков совпадают с p_nocnx.json W1 поле в поле. Smoke 190 (LAS 5,6 ГиБ): 210,7 с, пик RSS 4086 МиБ (весь конвейер), w5cf 46 с, 6205 объектов с признаками облака, пул 7120, записано 155. Флаг выкл: 3504/3506 как v3on (+3 новых файла кода). Удалены dist/koz3_v4off_055d3a5*.zip, koz3_v4off_65149fe*.zip, старый v4on_65149fe zip (манифесты оставлены). | собран, НЕ отправлен | /workspace/koz3/w4geo/dist/
01.11 22:45 MSK | W4 | v3_on regex/D-FINE: причина 1 regex cp312-only (9f7877b, vendor cp311–314 + purge), причина 2 huggingface_hub Kaggle 1.29 < 1.31 (b447649, vendor/_hf hf 2.2.0 только при несовместимом env); V3_DEGRADED.json+WARN, check_submission FAIL; архив v3on_b447649 sha 1406c01a…ef41d 3902 файлов DOC-1 OK; Kaggle v4: C/A/B FAIL (hf), NEG FAIL ок, D_env521=REF побайтно; v5 kernel koz3-w4-v3onchk запущен 22:30; npz numeric 1-D 10ee67b; удалены zip 10ee67b, v4on_da537d3 | 1406c01a | bots/w4-v3on c121b68
01.11 23:45 MSK | W4 | Kaggle v5 koz3-w4-v3onchk (архив v3on_b447649 sha 1406c01a…ef41d) COMPLETE: C_default PASS (FAIL 0, WARN 3), A_regex2025_preloaded PASS, B_no_regex PASS, D_new_env521 PASS — везде D-FINE из вендора 5.21.0, маркера V3_DEGRADED нет; smoke result.geojson 6b1da9bf…, ref103 result.geojson dc99ce1d… (n_det 155) побайтно = REF_old_ebc8e4ba_env521 во всех 4 средах; NEG_dfine_broken: маркер V3_DEGRADED.json есть, check_submission FAIL (5) — как задумано. Аренды dataset:aaaa0000004/koz3-w4-v3onfix и kernel:aaaa0000004/koz3-w4-v3onchk (bot-W4-v3fix) сняты | PASS, parity доказан | w4geo/kaggle/out_v3onfix2/
01.11 23:43 MSK | W4 | v2clean 545ef32a паритет: архив не пересобирается (b447649 не затрагивает: нет vendor/degraded.py, sp_dfine/gor_gate off); PREREG 8844d49; Kaggle koz3-w4-v2cleanchk2 v5 запущен 23:42 (REF_check4/C/A/B/D), лиза kernel до 02.11 03:41, опрос poll_v2clean_par.sh → out_v2clean_par | — | bots/w4-v3on
02.11 01:08 MSK | W4 | Kaggle v5 koz3-w4-v2cleanchk2 (архив v2clean 545ef32a, zip_sha_ok) COMPLETE 00:39: REF_check4 PASS (FAIL 0, WARN 2, 30 проверок), C_default / A_regex2025_preloaded / B_no_regex / D_new_env521 PASS (FAIL 0, WARN 2, 31), маркера V3_DEGRADED нет; smoke result.geojson 56be33ff…ca33 (58184 Б, 80 Polygon {finds_points 72, kurgany_tselye 3, kurgany_povrezhdennye 3, arkhitektury 2} = check4), ref103 result.geojson b7774f14…4ef5 (2639 Б) — побайтно = REF_check4 во всех 5 средах. По PREREG 8844d49 — PASS. Аренда kernel:aaaa0000004/koz3-w4-v2cleanchk2 (bot-W4-v2cpar) снята, опрос завершён | PASS, parity v2clean доказан | w4geo/kaggle/out_v2clean_par/
02.11 14:05 MSK | W4 | Q0 new34-диагностика: C1/A1/P1/P2 минус new34 целиком от перенастройки лидарных τ gor (gor_phys ф4, li_coarse ф5/7; спутниковая часть C1 +0.00075, A1 +0.00011, P 0); R1 −0.0023 спутниковый, большие сцены 035#s1 8.4 Мп −0.00157, 040#s0 5.7 Мп −0.0009 | Δ R1 −0.00232 | docs/bots/2026-11-02/bot-W4/q0_new34/Q0_NEW34.md
02.11 13:55 MSK | W4 | Q6 finds-GBM replay gate FAIL (UNI Δ−0.0075 P0.19; REP Δ−0.016), handed to W3 | 3b7a097 | docs/bots/2026-11-02/bot-W4/q6_finds_gbm/HANDOFF.md
02.11 14:12 MSK | W4 | C1F независимый регейт (PREREG e6b5a31, порт 98cdf56+фикс scene_drel для sp_dfine): реплей вложенно C1 Δall +0.01224 P 0.999, nolidar +0.0115, но new34 −0.0008 и kurgany_tselye −0.0033 < −0.002; P2 Δ +0.0153 new34 −0.0016 kt −0.0059; шкала dfine_f0 совпадает; v4 не собирался | FAIL | docs/bots/2026-11-02/bot-W4/c1f_regate/RESULT.md
02.11 14:30 MSK | W4 | C1F-gor/arkh + FP1 re-gate: PREREG + flag [post.w4_sat] (off) pushed, run started | d366a45, 07fa43c | docs/bots/2026-11-02/bot-W4/c1fx_regate/PREREG.md
02.11 14:33 MSK | W4 | τ-rule discrepancy CONFIRMED (old C1F rule changes 50/50 base τ, prod rule 0/50); C1F RESULT 5d0dd4c superseded | AMEND1 c60e50f | docs/bots/2026-11-02/bot-W4/c1fx_regate/PREREG_AMEND1.md
02.11 14:33 MSK | W4 | re-gate (prod rule, identity 0/50 OK): C1 Δ+0.016202 P1.0 PASS; GA Δ+0.013421 P1.0 PASS; FP1 Δ+0.008581 P0.999 PASS (=W1) | see RESULT | docs/bots/2026-11-02/bot-W4/c1fx_regate/RESULT.md
02.11 14:33 MSK | W4 | WRAP-UP (coord): v4 NOT built, no Kaggle; leases released | — | orch/bot-W4/STATUS.md
NEXT (recipe, W4): 1) re-gate is DONE: C1 (C1F all 5 classes), GA (gor/arkh, post hoc, P>=0.985714), FP1 (W1 as is) under prod τ rule (FP 0.5, metric._group_cuts midpoint, first argmax, round 6) with blocking identity sanity 0/50 — scripts docs/bots/2026-11-02/bot-W4/c1fx_regate/scripts (work copy /workspace/koz3/orch/bot-W4/work/c1fx, run_all.sh; code copy /workspace/koz3/w4/c1fx/x_c1fx = git archive of 07fa43c; outputs /workspace/koz3/w4/c1fx/out). Best single PASS = C1.
  2) v4: compute full-data C1 values (same rule, training = all regions: (d0,s) from grid + τ refit for 5 sp_dfine classes; add a "full" step to c1fx_prep.py), write [post.w4_sat] enabled=true {groups=["sp_dfine"], center={d0, s_by_class all 5}, thresholds_sp_dfine} into the v4 build config (v3_on.toml copy; note old W4 configs/v4_on.toml name collision), base v3_on b447649 + 98cdf56 port + 07fa43c flag; build via scripts/build_submission_git.py --check (DOC-1 guard, check_archive_clean, WEIGHTS_MANIFEST, ≤1 v4 zip, sha256) into /workspace/koz3/w4geo/dist.
  3) Kaggle parity C/A/B/D/REF + NEG via pool_kaggle.sh aaaa0000004, template /workspace/koz3/w4geo/kaggle/koz3-w4-v3onchk-fix, new dataset/kernel koz3-w4-v4*, is_private, leases dataset:/kernel:aaaa0000004/<name>, nohup poller (poll_v3onfix2.sh pattern). Push code only to bots/w4-v3on; no Drive.
  Commits: PREREG d366a45, flag 07fa43c, AMEND1 c60e50f, RESULT = this commit. Branch bots/w4-v3on.

30.10 14:52 MSK | C | v2-ref zip OK sha8f3f2671…; black-box REF emit RUNNING (docker pantum/n7q → w4geo/ref/out/REF_v2.geojson). MagGr DONE: C_* + R_d10_vsA18.json. Gate after REF. TerFl OFF. No aaaa/cpu-2.
- [2026-10-30 15:02 MSK] PIXGBM recover gate vs black-box REF_v2 (7194 feats, 18 train sites): **FAIL** (C_Cet and C_Setonly).
  C_Cet geo P/R 0.6318/0.4992 · C_Setonly 0.6233/0.4962 · REF 0.6644/0.7847 (v2_LB target 0.7694/0.6565). Main gap is MagGr recall: C 0.508 vs REF 0.859 (−0.351); EtGr −0.035.
  Honest Δ vs A18: Cet +0.0187, Setonly +0.0160 (surrogate ≥0.012 = True). TerFl flag was OFF; REF writes no terfl.
  Paths: runs/geo_pr_gate_{Cet,Setonly}_vsREF.{json,log}; REF w4geo/ref/out/REF_v2.geojson; needs EXPDS_METRIC=wt/w4-pix/materials/compute_metrics.py

30.10 15:09 MSK | C | MagGr FN-by-site → orch/bot-W4/REPORT_MAGGR_FN_BY_SITE.md. Worst ΔR: 201 (−0.82, C n=16), 211 (−0.59, C n=0!), 207/209. Idle-ready for next C_*; TerFl OFF; v3a frozen.

30.10 17:33 MSK | C | REF re-emit OK 7194. Gate CetO3_Sr8G FAIL vs REF (_ht MagGr R .681 vs .859). Surrogate +0.034. TerFl pix ON in Cet lineage (threshold_group=terfl_pix; oof maps included terfl). v3a frozen.

30.10 17:34 MSK | C | A ACK Sr8G FAIL. Idle for C*_noTerFl → re-gate vs REF + confirm 0× terfl_pix. Do NOT enable TerFl for Elena pkg. v3a frozen; no LB. REF_v2 already on disk (7194).

30.10 17:47 MSK | C | noTerFl gate: terfl_pix=0× OK. FAIL vs REF (see REPORT_TO_A_GATE_noTerFl.md). Soft-thr rejected. No LB.

30.10 17:48 MSK | C | A ACK noTerFl FAIL. Idle-ready: next C*=MagGr domain Гнёздово 211/212 (not soft-thr). Watch→19:40. No LB; v3a frozen.

## 30.10 23:04 MSK — FREEZE E1_G1 (Elena «как лучше», via A)
- Best cleanroom C*: `w4geo/work/cstar/C_CetO3_Sr8G_noTerFl_L_E1_G1.geojson` + `_ht` (sha c92b25e0 / aaba16ef).
- Gate vs REF_v2: FAIL PASS_vs_ref (MagGr R _ht wall −.162); geo+MagGr P _ht ≥ REF; honest Δ +0.0407/+0.0425; terfl_pix=0.
- Gate JSONs: orch/bot-W4/runs/gateC_CetO3_Sr8G_noTerFl_L_E1_G1{,_ht}.json
- STOP MagGr R grind vs REF. Hold package. No LB until Elena yes. C idle-ready for other W4 / seeds OOF when asked. v3a_min stays frozen.

01.11 11:10 MSK | C | STANDBY re-gate (A ORCH_PLAN_20261101). A on CPU PIXGBM P1. No GPU push. Wait A gate (B′ 0.315800). Hold 0.41238 + E1_G1 freeze. No LB.
01.11 11:39 MSK | C | Elena FULL GO GPU/CPU/TPU (A t225u). C ready for honest-10 / B′ gates on new lidar-aug + sat2 when A asks. STANDBY — no self kernel push. Hold 0.41238 + E1_G1.
01.11 18:22 MSK | C | RESUME STANDBY (A: NOT Elena STOP — quota cut cancelled turn by mistake). No GPU. Hold 0.41238 + E1_G1. Ready honest-10/B′ on A ask.

02.11 14:58 MSK | W4 (agent 5) | A→W4 ACK: v4 CONFIRMED. Zip /workspace/koz3/w4geo/dist/koz3_v4c1f_07795d2_51882efd_w6dfd2e07.zip sha256 f0def982… (manifest match), git 07795d2, [post.w1_c1f] enabled=true, DOC-1 name-scan clean (no geo_pixgbm/pixgbm/EN). Next: independent re-gate COMP_GORLOC + opt COMP_FL (coord-comp 5dedde3). Hold LB.

02.11 15:05 MSK | W4 (agent 5) | A ACK packaging: Drive v4c1f_f0def982 parts 00–07 uploaded + FINAL_CANDIDATES.md; Kaggle DS aaaa0000004/koz3-w4-v4c1f ready; kernel koz3-w4-v4chk RUNNING (parity 4+). FINAL_CANDIDATES/HANDOFF tip updated on box (wt/orch + orch/). GORLOC+FL re-gates continue. Hold LB (Elena via A).
02.11 15:10 MSK | W4 (agent 5) | COMP_FL re-gate (v4 honest-10, coord-comp 5dedde3 sat_frame_gate t2_cls, τ frozen): Δall +0.00012 P0.53, new34 +0.00056, nolidar −0.00283, kp floor −0.0027 FAIL → FAIL, |Δ|<0.003 → CLOSED. GORLOC: closed per A (parent regate_COMP_GORLOC_KMASK_CLOSE, W3 NO-PASS). runs/regate_COMP_FL.json. No LB.
02.11 15:20 MSK | W4 (agent 5) | Гейт C1FP (C1F-all+FP1) на v4 → v5: FAIL. vs base Δall +0.01344 P1.0, new34 +0.0019, nolidar +0.0212, но пол kt −0.00227 (<−0.002); vs v4(C1) Δall −0.00276 P0.037, nolidar −0.0119. v5 не формируется, v4 остаётся. runs/gate_C1FP_v5.{json,md}, out sha d8440375…. Hold LB.
