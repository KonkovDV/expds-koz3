# REPORT_v3 — кандидат v3 бота 7 (29.10.2026, МСК)

## English summary (updated 29.10 ~12:10 MSK)
- **Branch** `bot7-candidate-v3` (local, no push), HEAD **c1a6b7e** (docs-only on top of 36af6b7; zips built from 36af6b7, docs are not packaged, so no rebuild). v2 zip `work5/dist/submission_bot7cand_v2_eb268d6.zip` (8f3f2671…) untouched.
- **FR1h ON** (frame_prior @d4edf2f; groups sp_or/sp_focal/sp_focal2; all classes except gorodishcha; d0 0.5, edge 0.5), before thresholds in both finish and safe_finish. Bot-10: TOFR−TO +0.004042, P 0.999, CI90 [+0.00126; +0.00910], LOFO +0.004228, 5/5; gorodishcha unchanged. Scene boxes identical to bot-10's on 168/168 pairs. Caveat: package keeps v2 sp_* thresholds (bot-10 used fold thresholds on FR1h curves for evaluation only) — QUESTIONS.md.
- **D-FINE: the submission model is f0.** `final` is RESERVE only; switching needs quantile-transferred thresholds (or honest recalibration) and a separate bot-10 acceptance — NOT a one-line switch (`work6/on/FINAL_VARIANT.md`; make_on.py refuses non-f0 weights without a thresholds JSON). fp32 weights + autocast, no `.half()` (test).
- **check_submission** now hard-FAILs when sp_dfine is enabled and the zip's weights give 0 queries ≥0.3 on 12 reference windows of 103 SpOr (f0 gives 14, max 0.704), and when the smoke log has `sp_dfine n_det=0` on a known region. 4 new tests + FR1h tests (3).
- **Zips** (`work6/dist/SHA256SUMS`):
  - submission: `submission_bot7cand_v3on_36af6b7.zip` 9ec25962ed70ba258c5c4156f443198072c48fe5449e5cc9aa10d8f7e3bd74e9 (FR1h on, D-FINE f0 R, others off). check_submission --no-smoke PASS (FAIL 0, WARN 1 vendor .py), weights probe 14/14, max 0.7037.
  - flags-off: `submission_bot7cand_v3off_36af6b7.zip` 4d4107d1689488fc50742cc139c79582541b1c0928eafee0ab9b45a7d3bbc2f9 (FR1h on, D-FINE off).
  - kept: v3off_8857343, v2 eb268d6. Deleted: 1d6c822 and 1cefcee zips.
- **pytest 1cefcee:** 451 passed, 9 skipped, 1 failed — `test_coco_roundtrip` (no `cv2` in the box venv; training-only script; earlier runs deselected it).
- **ORSP (abae911) on 103, base = v3off_36af6b7:** povr 0.2665 and 0.3645 give identical outputs and score Δ 0.0 (ALL 0.006993, sp_or 0.0952). Final kurgans: sp_or povr 1 (conf 0.4781 → 0.5399 fused), sp_focal2 0 in all three. One region — not evidence. Under D-FINE R the sp_* groups are dropped, so ORSP is a no-op there.
- **Graceful degradation (`work6/tt`, 18 regions):** full budget — new (FR1h block stripped) vs old 8857343 **byte-identical**, 184 objects, 647 s vs 732 s, peak RSS 1774 / 1866 MB. Cut budget (budget 200 s, soft 150 s, optics_s_per_gb 400 — artificial stress): both exit 0, valid GeoJSON, inside budget (156 s / 150 s), RSS 1576 / 1344 MB. But new_cut 6 objects (li 2, sp_or 4) vs old_cut 14 (ae 1, li 2, sp_or 11): the new code cut the rest of lidar to reserve optics, and optics still got no time (geophysics + lidar slowed by box load: 9 s + 36 s vs 2 s + 8 s). No crash, but under this stress there is no gain. Not triggered at the real budget.
- **Data rule 29.10 (§7, docs/THIRD_PARTY.md):** no Yandex, Google, Bing or Esri; no previous-stage data; FR1h and the audit modules use no external data. Doubtful: ImageNet init (all variants, incl. v2), COCO and HGNetV2-SSLD in D-FINE f0 (v3on only); DINOv2 LVD-142M not packaged; bot-9 obj2coco (Objects365) not used and must not be.
- **Blockers:** 4090 time/VRAM; valid PC e2e re-run with v3on_36af6b7; FR1h threshold question; import inside pantum/n7q image; **organizers' answer on ImageNet/COCO under the explicit-ML-licence rule** (if no — use v3off; ImageNet also affects v2).

---

## 1. Что в v3 (ветка `bot7-candidate-v3`, без push)
Цепочка от v2 eb268d6: 4327c87 → f59f016 → d9c7ae3 → cc854b3 → 8ec6a88 → 11f8658 → **8857343** (принят ботом 10 условно) → e09de12 (bot-6 315dc91) → eda980b (bot-6 c050028 ORSP) → 05c719f (путь весов D-FINE в конфиге; смена модели — НЕ одной строкой, см. §3) → **9138bff** (шлюз bot-5) → b5c51b9, c0790b4, 8f89294 (аудит) → 4f03a3e (ORSP, выкл.) → 1295c75, 1d6c822, 85aee5e (починка D-FINE) → 1cefcee (FR1h ВКЛ., проба весов D-FINE) → **36af6b7** (комментарии конфига, HEAD).

| Флаг | Модуль | По умолчанию | Статус приёмки |
|---|---|---|---|
| `[sp_dfine]` | `expds/sp_dfine.py` (порт bot-9, без хаба) | выкл. | bot-10 REPORT10_accept §1: ACCEPT R/C условно (4090, e2e, финальные веса) |
| `[gor_gate]` | `expds/gor_gate.py` (порт bot-5 gor_v3) | выкл. | ждёт приёмки bot-10; только без R |
| `[post] kurgan_in_gor` | `expds/post.py` | выкл. | отклонено мной: −0,000848, P 0,187 (`work6/kig`) |
| `[post.frame_prior]` (FR1h, d4edf2f) | `expds/post.py` | **вкл.** | bot-10 C3h: TOFR−TO +0,004042, P 0,999, LOFO +0,004228, 5/5, городища без изменений → PACKAGE-ONLY, включено |
| `[post] sp_score_fusion` (ORSP) | `expds/post.py` | выкл. | bot-10 §3: честно +0,002523, +0,003008 — утечка порога; не настраивал |

Модули за выключенным флагом в архив не идут (8857343): иначе новый check_submission даёт FAIL imports (transformers).

## 2. Шлюз городищ DINOv2-S (бот 5) — интеграция (9138bff)
**Источник:** `bot-5/FOR_BOT7_GOR_V3.md`, ветка `bot5-gor-patch-kaggle` abb89d0 + 82058b8 (`bot-5/repo`).

**Артефакты и md5 (проверено мной):**
- бэкбон `facebook/dinov2-small` @ `ed25f3a31f01632728cabb09d1542f84ab7b0056`, `model.safetensors` md5 **ea251be6e151ba8609ad0bc9b2e13a78** (совпало), 22 056 576 параметров F32; `config.json`. Источник на box: `bot-5/tmp/hf/hub/models--facebook--dinov2-small/snapshots/ed25f…/`.
- голова `bot-5/analysis/gor_patch/gor_v3_head.npz` md5 **acb95c52956d712a16c3dd389237a691** (совпало): mu, sd, W (1536×64), qsd, w (65); p = σ([1, ((F−mu)/sd)·W/qsd]·w).
- md5 обоих файлов сверяется при загрузке в пакете (`backbone_md5`, `head_md5` в DEFAULTS); при несовпадении шлюз не применяется (лог), выдача — как без шлюза.

**Лицензия:** DINOv2-S — Apache-2.0 (код и веса; API HF на этой ревизии: `cardData.license = apache-2.0`, проверено 29.10). Предобучение — LVD-142M (Meta). Голова — своя (bot-5), только трейн КОЗ-3 SpOr. Записано в `docs/THIRD_PARTY.md`. DINOv3 не используется.

**Порт (`expds/gor_gate.py`):**
- признаки — точный порт `extract_feats` bot-5: виньетки 384 м шагом 192 м + контекст 768 м, 224 пкс, растяжка 2/98 перцентиля по уменьшенной копии, маска alpha (полоса 4) или nodata, валидность ≥ 0,5; [CLS, mean] на масштаб → 1536-d; батч 16.
- загрузка без `from_pretrained`/хаба: `Dinov2Config.from_json_file` → `Dinov2Model` → `load_state_dict(strict)` с переименованием ключей внимания 4.x → 5.x (`attention.attention.query/key/value` → `q_proj/k_proj/v_proj`, `attention.output.dense` → `o_proj`); пропущенных и лишних ключей 0. Без переименования веса не грузились бы (мой первый прогон упал на проверке ключей — проверка сработала как надо).
- исправлен недочёт bot-5: `d.colorinterp[3]` при < 4 полос → IndexError; у меня проверка `d.count ≥ 4`.
- p_scene = max p по виньеткам всех SpOr-растров участка; conf городищ sp_or (без threshold_group) × p_scene → `threshold_group = "sp_or_gated"`, порог `[thresholds_by_markup.sp_or_gated] gorodishcha = 0.992648`.
- участок гейтится только если все его растры пройдены; ошибка, нехватка времени (`guard.ok(30)` и мягкий потолок `_soft_ok(guard, "gor_gate", …)`) или отсутствие артефактов → участок без изменений (= v2).
- если целей нет (вариант R выбросил sp_or) — бэкбон не грузится, 0 с.
- сборка (`build_submission.py`): при `[gor_gate] enabled` кладёт `models/dinov2-small/{config.json,model.safetensors}`, голову и вендор `inference/lib/vendor` (transformers 5.21.0 из колёс bot-9; нужен и шлюзу).

**Сверка порта с bot-5** (`work6/gate/port_check.{py,json}`, CPU 2 потока, taskset 0,1, nice): 
| Участок | виньеток (порт / bot-5) | max\|ΔF\| (fp16) | min cos | max\|Δp\| | p_scene | с/виньетку |
|---|---|---|---|---|---|---|
| 046_Тыва_1 | 4 / 4 | 0,0 | 0,99999988 | 2,3e-9 | 0,0103 | 1,47 (прогрев) |
| 185_Изборск | 47 / 47 | 0,0 | 0,99999988 | 1e-16 | 0,9996 | 0,47 |
| 103_Гнёздово_Ольшанская | 29 / 29 | 0,0 | 0,99999988 | 0,0 | 0,0730 | 0,41 |
Пик RSS 830 МБ. Сквозной `gor_gate.apply` на 046+185 с артефактами по путям пакета (`work6/gate/apply_check.{py,out}`): 32 с вместе с загрузкой и md5 (≈8 с); городища получили sp_or_gated и conf × p, курганы не тронуты.

**Время против 1200 с:**
- CPU 2 потока: 0,41–0,47 с/виньетку с чтением (у bot-5 — 0,13–0,16; разница, вероятно, чтение/ЦП box). В среднем 32,6 виньетки на участок → ≈15 с на участок; 3982 виньетки трейна → ≈30 мин. **На CPU при полном тесте шлюз в 1200 с не влезает** — сработает мягкий потолок, часть участков останется без шлюза (= v2 на них).
- GPU: замера нет. Оценка bot-5 10–40 мс/виньетку на T4 → ≈1–3 мин на 3300–4000 виньеток, упор в чтение. Нужен замер на 4090 (не мне).
- Под R шлюз 0 с (целей нет).

**Честность порога (ответ на «LOFO вложенный?»):**
- В honest-10 p бралось вне фолда (5 семейных фолдов, кандидатская семья не видна, `scores_cand7.json`), пороги — вложенно через `honest_counts` (sweep_groups вне фолда). Значит +0,005057 и LOFO +0,002423 — честные числа процедуры.
- Гиперпараметры (PCA-64, C=1, чистка 0,85/0,05) заданы заранее (FOR_BOT7_GOR_V3.md).
- Развёрнутый порог 0,992648 подобран `sweep_groups` на всём трейне по OOF-p, а развёрнутая голова обучена на всех данных — **в выборке**, как все пороги v2. Тестовых семей бот 5 не видел (тест скрыт); данных предыдущего этапа нет (только train-expds-3 SpOr).
- Оговорка: голова на всех данных даёт на трейне p выше OOF (in-sample p_scene на 185 = 0,9996), так что порог по OOF-p может быть немного мягким для финальной головы.
- Прирост ниже MDE; при R шлюз бессмыслен. **Рекомендация: в сдаче шлюз выкл.**; включать только если D-FINE R не идёт.

## 3. D-FINE SpOr (бот 9)
- Порт и проверки — `work6/FOR_BOT9.md`. Сверка выхода с bot-9 330/330 побитово (`bot-9/submission_dfine/smoke/smoke.json`).
- Лицензии: `ustc-community/dfine-large-coco` @ aa8f8211… Apache-2.0, только COCO (Objects365 нет); бэкбон HGNetV2-B4 ImageNet. ResNet-18 ImageNet (наши U-Net) — torchvision BSD-3, разрешено 29.10.
- Веса f0: `/workspace/kaggle_out/koz3-dfine-spor-full-rest/dfine_f0`, sha256 81ef8228…, 125 001 980 Б fp32. **Модель сдачи — f0** (решение бота 10). `final` (sha256 bdb579ca…3050729) — ТОЛЬКО РЕЗЕРВ: у final нет OOF и сдвинуто распределение уверенностей, поэтому замена пути весов одной строкой НЕДОПУСТИМА — нужен перенос порогов по квантилям (или честная перекалибровка) и отдельная приёмка бота 10 (`work6/on/FINAL_VARIANT.md`; make_on.py без JSON порогов для не-f0 весов завершается с ошибкой). Храним fp32, на cuda только `torch.autocast`, без `.half()` (тест `test_sp_dfine_fp32_weights_autocast_no_half`).
- Время/VRAM 4090: **замера нет**. Выход ядра `koz3-dfine-spor-final` в `/workspace/kaggle_out` отсутствует (есть только -full, -full-rest, -smoke). Оценки bot-9: 70–110 с/фолд с потоковым чтением, 260–330 с с последовательным (наш порт — последовательный); VRAM инференса fp16 bs8 2,0–2,1 ГБ + 0,3–0,5 ГБ контекста (DFINE_SUBMISSION.md:48).
- ~~Рекомендация: финальная модель~~ — заменено решением бота 10 (29.10 10:46): в сдаче f0, пороги sp_dfine подобраны по OOF фолдовых моделей (f0-семейство); final — резерв (см. выше).
- **Проверка весов в check_submission (36af6b7):** при `[sp_dfine] enabled` веса из архива прогоняются на 12 опорных окнах 103_Гнёздово_Ольшанская SpOr (CPU, batch 1, без flip); FAIL, если запросов с score ≥ 0,3 ноль или модели не загрузились. f0: 14 запросов, max 0,704; v3on_36af6b7: 14, max 0,7037 — PASS. В смоук-логе `sp_dfine n_det=0` на известном регионе — FAIL. Прогон PC e2e засчитывается только при n_det>0, без SP_DFINE_UNAVAILABLE и при отличии выхода v3 от v2 — pc_e2e_v3.ps1 ред. 2 это проверяет (подтверждено).

## 4. Проверки
- pytest HEAD 9138bff (`-k "not coco_roundtrip"`, 2 потока, taskset 0,1, nice): **412 passed, 9 skipped, 3 deselected**, 390,7 с (`work6/pytest_gate.log`). Новые тесты: голова/md5, «нет целей → бэкбон не грузится».
- check_submission + побайтная сверка с eb268d6 (флаги выкл.):
  - 4f03a3e: synth PASS 153 s, 210 PASS 337 s, 214 PASS 212 s, 217 PASS 624 s (the slowdown was concurrent load on the box), all BYTE-IDENTICAL with eb268d6 (`work6/chk_4f03a3e.out`). Zip 4f03a3e is deleted (superseded by 1d6c822).
  - Time protection is ON by default (8f89294) and the output is byte-identical because `_restore_optics_order` restores the object order. The graceful-degradation test (`work6/tt/tt.sh`, `tt/tt.log`, 29.10 11:39) **has been run**. Full budget: new vs 8857343 byte-identical (184 objects). Cut budget: both valid and inside 200 s; new 6 objects vs old 14 (lidar cut for optics, optics still empty). See the summary.
  - gormask, neigh, calib and frame stay OFF per bot-10 REPORT11:
    - calib: REJECT, −0.004183;
    - gormask: REJECT, −0.005630;
    - frame: +0.008344, P 0.997, LOFO +0.006542, 5/5, but fails П4 (new34 −0.003728, 7 extra FP gorodishcha on lidar sites), so not ACCEPT;
    - neigh, C2 and the PIXGBM calibration shift: PENDING.
  - Our kurgan_in_gor (−0.000848) is replaced by the audit gormask, which is also OFF.
- W2 (zip 8857343, вход 217, taskset 0,1; `work6/w2/w2.json`): procs=1 — 162,8 с, пик дерева 626 МиБ; procs=2 — 125,8 с, пик дерева 1463 МиБ, макс. процесс 610 МиБ. **В дереве 3 процесса** (родитель + 2 воркера PIXGBM) — по букве правила «≤2 процесса» это пограничный случай; выход идентичен v2. v2 не ограничивает потоки PIXGBM (замечание бота 4).
- PIXGBM RSS по замеру бота 3:

  | procs | RSS | время |
  |---|---|---|
  | 1 | 492 МБ | 43 с |
  | 2 | 1013 МБ | 37 с |
  | 4 | 1505 МБ | 36 с |

  Это ≈ +0,25 ГБ на каждый дочерний процесс. Потолок потоков PIXGBM — 2 (8f89294).
- ORSP abae911 (cherry-pick 4f03a3e) выключен. Проверки (a)–(e) на 103 (`work6/orsp/orsp_checks_p2665.txt`, `orsp_checks_p3645.txt`), база = v3off_36af6b7 (FR1h вкл.), флаг вкл. с порогом sp_focal povr 0,2665015 и 0,3645:
  | вариант | ALL | sp_or | курганы sp_or в выдаче | курганы sp_focal2 в выдаче |
  |---|---|---|---|---|
  | база (выкл.) | 0,006993 | 0,095238 | povr 1 (conf 0,4781) | 0 |
  | вкл., povr 0,2665 | 0,006993 (Δ 0) | 0,095238 (Δ 0) | povr 1 (conf 0,5399 после слияния) | 0 |
  | вкл., povr 0,3645 | 0,006993 (Δ 0) | 0,095238 (Δ 0) | povr 1 (0,5399) | 0 |
  - Выходы 0,2665 и 0,3645 побайтно равны: после слияния вторая по величине уверенность sp-кургана 0,1965 < 0,2665.
  - До порогов при вкл. (gen_floor 0,05): sp_or povr 336 / tselye 6, sp_focal2 povr 74 / tselye 7; слияние 424→423 sp-курганов, sp_partner_only 1 → 0 в выходе (c); проверка кропа уверенность не меняет (b); merge_within_markup одинаков (d).
  - (e) Один участок — не доказательство; включать только по честному замеру бота 10 (Δ>0, P≥0,85).
  - Взаимодействие с D-FINE R: R удаляет группы sp_or/sp_focal/sp_arch/sp_focal2 на полностью пройденных SpOr-участках, поэтому там ORSP (и FR1h) ничего не делают; действуют только на участках, где D-FINE не прошёл (мягкий потолок, ошибка → v2).
- FR1h (1cefcee): шаг в watchdog.finish и safe_finish между kurgan_neighbour_prior и sp_score_fusion/apply_thresholds (тест `test_fr1h_before_thresholds_in_finish`); городища не трогаются (тест); рамки сцен = bot-10 frame_boxes.json на 168/168 парах (`work6/fr1h/boxes_cmp.out`). На 103 выход v3off_36af6b7 по баллу = v2 (ALL 0,006993, sp_or 0,0952). Оговорка: пороги sp_* в пакете — v2; bot-10 мерил с фолдовыми порогами на кривых FR1h (медианы k10 sp_focal povr 0,3608 / tselye 0,3361, sp_focal2 0,3798 / 0,5723 против пакетных 0,3645/0,3725 и 0,501645/0,645505) — вопрос боту 10 (QUESTIONS.md).
- pytest 1cefcee (`work6/pytest_1cefcee.log`): 451 passed, 9 skipped, 1 failed — `test_coco_roundtrip` (в venv box нет `cv2`; скрипт только для обучения, раньше исключался `-k "not coco_roundtrip"`). Целевые 21 тест D-FINE/FR1h — зелёные.
- Данные предыдущего этапа (КОЗ №2) не использовались ни для подбора, ни для обучения.
- Флаги ВКЛ. (тестовые архивы, не для сдачи):
  - **D-FINE load fix (1295c75, 1d6c822, 85aee5e).** Bot 1's PC e2e was invalid because the Linux vendor does not load on Windows Python (details in `work6/FOR_BOT1_e2e.md`). Changes:
    1. `import_dfine`: purge, vendor first, environment only at ==5.21.0.
    2. ERROR `SP_DFINE_UNAVAILABLE` / `GOR_GATE_UNAVAILABLE`, then v2 output.
    3. `sp_dfine n_det` logged per region and in total.
    4. check_submission: FAIL on the markers, on a missing vendor or a version ≠ 5.21.0, and on a failed clean-process probe. HF weight directories no longer count as missing, and vendored packages are INFO instead of an imports FAIL.
    5. With only the gate enabled, `sp_dfine.py` is packed too.

    Tests: `tests/test_bot7_dfine_import.py`, 14 tests. Full pytest on 1d6c822: 440 passed, 9 skipped, 3 deselected (`work6/pytest_dfine.log`).
  - `submission_bot7cand_v3on_1d6c822.zip` (R, f0, in-sample thresholds): check_submission `--no-smoke` gives PASS (FAIL 0, WARN 1 for extra .py files in the vendor), with the dfine probe importing from the vendor (`work6/check_v3on_1d6c822_static.log`).
    - With smoke, the run hits the box's 2 GB RSS cap on CPU fp32 at batch 8 (`check_v3on_1d6c822_103.log`). Before that cap, the log shows `transformers 5.21.0 из vendor` and "моделей 1".
    - Run of main.py at batch=1 (a test-only override in an unpacked copy, not in the zip), Linux py3.12 CPU, 2 threads, region 103 (`work6/on_b1_103.log`): `sp_dfine n_det=154 total`, drop_groups −3, 220.7 s (v2: 139.4 s), peak RSS 1557 MB.
    - e2e 103 (`work6/e2e/e2e_103_1d6c822.json`): ALL v2 0.006993 vs v3on 0.0; sp_or 0.0952 (3 objects) vs 0.0 (5 objects), against 14 GT objects. **This is one region and one scene on CPU without fp16, so it is not evidence.** The comparison has to come from bot 10's folds and the PC run on 101+103.

## 5. Честность приростов (сводка)
| Что | Число | Чьё | Честно? |
|---|---|---|---|
| D-FINE R − A | +0,019802, P 0,976, ДИ [+0,00015; +0,04612], 5/5 | bot-10 | пороги вложенные (honest_counts); выбор R среди C/E/R на тех же данных (R−C +0,000005 — смещение мало); статичный стек; пакетные пороги в выборке |
| Шлюз − A | +0,005057, LOFO +0,002423, 5/5 | bot-5, проверка bot-10 нужна | p вне фолда, пороги вложенные; ниже MDE; при R = 0 |
| ORSP | +0,002523 честно (+0,005531 с утечкой +0,003008) | bot-10 | флаг выкл., не настраивал |
| KIG | −0,000848, P 0,187 | я | отклонено |
| флаги выкл. | 0 (побайтно = v2) | я | — |

## 6. Блокеры
1. Нет замера 4090 (время фаз, пик VRAM) — ядро `koz3-dfine-spor-final` бота 9 (GPU не мне).
2. (снят) В сдаче f0; final — резерв, не одной строкой (перенос порогов + приёмка).
3. Сквозной прогон «v3 вкл. vs v2»: прогон бота 1 на ПК НЕДЕЙСТВИТЕЛЕН (D-FINE не загрузился на Windows). Нужен повтор с `submission_bot7cand_v3on_36af6b7.zip` и `work6/pc/pc_e2e_v3.ps1` ред. 2 (предпочтительно `-Mode docker`). Скрипт сам проверяет, действителен ли прогон.
4. Чтение D-FINE последовательное (bot-9 `detect_tif` с потоками не подключён — жду сверки bot-9).
5. robust_accept для R не считается (нет кривых c5_R/c5_A).
6. (снят) Аудит интегрирован: b5c51b9, c0790b4, 8f89294. Тест деградации прогнан 29.10: падений нет. При искусственном урезании новый порядок дал 6 объектов против 14 — оптике выигрыша нет. Порог `optics_s_per_gb` стоит перепроверить на ПК.
12. **Правило о данных 29.10 (§7):** нужен ответ организаторов, допустимы ли ImageNet-инициализация (все варианты, включая v2) и предобучение D-FINE на COCO (только v3on) при требовании «явной ML-лицензии». Решает бот 1.
10. Импорт вендора внутри самого образа pantum/n7q не проверен (на box Linux py3.12 проходит). Закрывает повтор на ПК в режиме docker.
11. Смоук check_submission с D-FINE на CPU box не укладывается в 2 ГБ при batch 8. На GPU площадки это не ограничение (96 ГБ), но полный смоук флагов ВКЛ. возможен только на ПК или GPU.
7. Пороги sp_dfine и sp_or_gated в пакете — в выборке.
8. Шлюз на CPU не влезает в 1200 с на полном тесте (мягкий потолок спасает, но шлюз частичный).
9. ORSP: утечка порога — ждём bot-6; safe_finish без шага fusion.

## 7. Правила о данных (RULES_ORG_2026-10-29_data.md) — аудит 29.10

Подробно — `docs/THIRD_PARTY.md` (c1a6b7e), раздел «Правило организаторов о данных».

- **Нарушений не найдено.** Яндекс Карт, снимков Google, Bing, Esri и их производных в коде и данных нет. Данных прошлого этапа (КОЗ №2, `ackrome/satelite-scan`) нет: датасет не скачивался.
- **FR1h** использует только геометрию сцен входных растров. Идея — из OSINT публичного репозитория соперника №9 (взяты идеи, кода нет). d0 и edge измерены на OOF трейна КОЗ-3.
- **Модули аудита** — правила и множители над нашими кандидатами, данных извне нет.
- **Свои веса** — трейн КОЗ-3. В .pt не записаны пути к данным; вывод сделан по скриптам и протоколу.
- **D-FINE f0** (v3on): `ustc-community/dfine-large-coco@aa8f8211`, Apache-2.0, COCO train2017. Бэкбон HGNetV2-B4 на ImageNet (SSLD). Дообучение — бакет `train-expds-3`. Objects365 не используется.
- **Сомнительно, явно:**
  - (1) ImageNet-1k — условия «non-commercial research»; затрагивает v2, v3off и v3on;
  - (2) COCO — изображения Flickr под смешанными лицензиями; только v3on;
  - (3) SSLD мог использовать дополнительные изображения — не проверено;
  - (4) DINOv2 / LVD-142M — не в пакете;
  - (5) у bot-9 ядра obj2coco (Objects365, по каталогам некоммерческое) — в наш пакет не входят, брать нельзя;
  - (6) ключи доступа к `train-expds-3` в ядрах bot-9 — происхождение не проверял, ключи не читал.
- Если правило читается строго: v3off убирает (2)–(3); (1) остаётся во всех вариантах.
