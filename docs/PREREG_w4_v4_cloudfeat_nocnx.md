# PREREG W4 v4 nocnx — RR1s-cf без cnx_* (01.11 21:05 MSK, до расчёта)

## Почему новый вариант
Модуль W1 `expds.w5cf_infer` (bots/w1-cfdeploy b62f59e) на инференсе не имеет 5 признаков `cnx_*`
(ConvNeXt-лидар W2, два сида). В архиве v3_on ConvNeXt-лидара нет (models/: только r18-сети li_*/spor_*/surf/ae,
D-FINE, pix_clean, LGBM; группа `li_cnx` в v3_on.toml закомментирована) → cnx_* на инференсе посчитать нельзя,
вариант J2/8a0d48d (с cnx) не разворачиваем. Один новый вариант: **V4N = V3P + RR1s-cf nocnx** (ранжировщики
обучены на X4 с cnx = NaN). Выбран после семейства J2 → Бонферрони сохраняется.

## Данные и движок
Как в docs/PREREG_w4_v4_cloudfeat.md / PREREG_w4_regate_W1J2.md: V3P (table.pkl, C19 etonly orget, D-FINE),
FINAL-конвейер и полы, honest-10 family_folds (сверка с ws_rr), без M_rp. Скоры ранжировщика: outer/inner
из `/workspace/koz3/w1/cf/w5d/work/nested_w5cfnc.npz` (W1, nested2.py --xset X4nc). Бленд тот же
(S = α·rt + (1−α)·RB для pool-строк li, RB = CDF conf li по классу).

## Гейт (решающий)
* α_k вложенно по фолдам из {0.3, 0.6, 1.0} (максимум inner honest), пороги honest-10.
* V4N против V3P, бутстрап 2000 по семьям. PASS = Δall ≥ +0.003 **и** P ≥ 0.9857 **и** полы классов
  (−0.005, тесные −0.002) **и** new34 ≥ 0 **и** nolidar ≥ 0.
* Вывод: runs/independent_regate_W1J2_nocnx.json.

## Deploy-параметры (подгонка, после гейта, на результат гейта не влияют)
* α = мода вложенных выборов V4N (при равенстве — меньшее α).
* Пороги: один sweep_groups по всем регионам трейна на V3P + RR1s-cf nocnx(α, outer OOF); в v4_on.toml —
  только группы с li-строками (li, li_dev, li_mosaic, li_coarse, li_fort, li_mask, gor_phys); прочие = v3_on.
* ref (conf_li, rb_pool, r_pool по классам; r_pool из outer nested_w5cfnc) → models/w5cf/ref.npz.

## Проверка через архив (после сборки v4_on)
Скоры модуля на трейн-регионах через код архива против nested-оценки: ожидается та же величина Δ
(модели на всём трейне ≠ OOF; расхождение сообщается, не перевыбирается).
