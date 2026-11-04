# PREREG W4 v4 (w5_cloudfeat) — deploy-параметры и оценка состава v4 (01.11, до расчёта)

Состав v4 = v3_on + RR1s-cf (J2 **без** M_rp) за флагом `w5_cloudfeat` (по умолчанию false).
Данные/движок те же, что в docs/PREREG_w4_regate_W1J2.md (V3P, table.pkl, nested_w5cf.npz W1, honest-10 family_folds).

## 1. Deploy-параметры (подгонка, не замер)
* α = мода вложенного выбора J2 (runs/independent_regate_W1J2.json, picks) = **0.6** (7 из 10 фолдов).
* Шкала: для li-строк класса c score S = α·rt + (1−α)·RB, где RB = эмпирическая CDF conf li-строк класса c
  на train-таблице (mid-rank, как rankdata/n − 0.5/n), rt = квантиль RB по pool-строкам на уровне CDF score
  ранжировщика среди pool-строк (outer OOF W1). Не-pool li-строки: S = RB. Референсные таблицы
  (отсортированные conf_li, RB_pool, R_pool по классам) → `models/w5cf/ref.npz` (без меток, без координат).
* Пороги: один sweep_groups по всем 110 регионам (все фолды) на кривых V3P + RR1s-cf(α=0.6, outer OOF),
  без M_rp. В v4_on.toml переносятся пороги групп, где есть li-строки (li, li_dev, li_mosaic, li_coarse,
  li_fort, li_mask, gor_phys); прочие группы — как в v3_on.

## 2. Оценка состава v4 (информативно; решение — за родителем)
* V4C = V3P + RR1s-cf без M_rp, α_k выбирается вложенно по фолдам из {0.3, 0.6, 1.0} (как J2), пороги — honest-10.
* Сравнение с V3P, бутстрап 2000 по семьям. Критерии те же: Δ ≥ +0.003, P ≥ 0.9857, полы классов, new34 ≥ 0, nolidar ≥ 0.
* Ожидание: ≈ Δ(J2) − вклад M_rp (J2 +0.0124; M_rp на V3P ≈ +0.0028).
