# REPORT13 — задача B: C2 (ансамбль pixgbm) и SATf, приёмка П1–П6 на honest-10
bot-10, 2026-10-29 (МСК). PREREG: `PREREG_C2_SATf.md` (mtime 11:15:14; примечание 11:55) поверх `bot-3/audit/PREREG_C2.md` и
`PREREG_SATf.md` (sha256 в примечании). Метрика — `repo/materials/compute_metrics.py`; `work4/paired.py`, 2000 бутстрепов по семьям.
Данные: `bot-3/audit/calib/pred_ens2.geojson` (C2 = аналог ens2, 2 модели; не 3-модельная сдача), `bot-3/audit/var/SATf/pix_lofo.geojson`,
`bot-3/pixgbm/pix_lofo.geojson` (S). Конвейер `work13/run13b.sh`.

## Результаты k10 (keep — основной)
| сравнение | keep Δ | P | случ. раскл. | drop Δ | источник |
|---|---|---|---|---|---|
| C2 − S (global) | **−0.002733** | 0.118 | 1/5 | +0.002007 (P 0.569) | work13/paired_g_vsS.log |
| C2 − S (site) | −0.003269 | 0.093 | 1/5 | +0.003536 (P 0.518) | work13/paired_s_vsS.log |
| C2s6 − Ss6 (global) | **−0.027693** | 0.000 | 0/5 | −0.021478 | work13/paired_g_vsSs6.log |
| C2 − A (справочно) | +0.009202 | 0.756 | 5/5 | +0.012987 (P 0.667) | work13/paired_g_vsA.log |
| C2s6 − A (справочно) | −0.014223 | 0.082 | 1/5 | −0.010438 | work13/paired_g_vsA.log |
| SATf − S | +0.004016 | 0.629 | — | — | work13/paired_g_vsS.log |
| SATf − S (site) | +0.004131 | 0.728 | 5/5 | +0.004131 | work13/paired_s_vsS.log |
| SATfs6 − Ss6 | +0.003082 | 0.629 | — | — | work13/paired_g_vsSs6.log |
| SATf − A (справочно) | +0.015951 | 0.948 | 5/5 | +0.014995 | work13/paired_g_vsA.log |

(S − A = +0.011935 keep, P 0.948 и Ss6 − A = +0.013470 — те же, что в принятии S; P 0.948 — вырожденный бутстреп, меняются 3 семьи.)

## Вердикты
- **C2: REJECT.** К своей базе S: Δ < 0 в keep (−0.0027 global / −0.0033 site), П1 и П2 (P 0.118 / 0.093) и П3 (1/5) не выполнены;
  со стеком s6 — сильно хуже (−0.0277, P 0). В drop Δ +0.002, но P 0.569 < 0.85 — тоже не проходит П2. Плюс к A (+0.0092) целиком из S, сам ансамбль не добавляет.
- **Время:** mock_val S 150.7 с против C2 318.4 с (×2.11, 2 потока; `bot-3/audit/FOLLOWUP_REPORT.md` l.59–61, 78); prob_map ×2.44–2.67
  (`bot-3/audit/out/ens_timing.json`). Цена времени без выигрыша балла — ещё один довод против.
- **SATf: REJECT** по предрегистрации bot-3 (П2 не выполнен: P 0.649 global / 0.742 site, `bot-3/audit/FOLLOWUP_REPORT.md` п.3); не
  перемерялось по решению bot-1. Наш k10 Δ к S +0.004016 (P 0.629) совпадает с их +0.0040 и тоже не проходит П2/П5.
- Под флагом EtGr-слоёв (REPORT13_etgr_layers.md) вся ветка pix-геофизики теряет выигрыш по EtGr — дополнительно против C2/SATf.
- Вторичная проверка rw (крупные сцены) и site для s6/к A не досчитаны — сняты по бюджету в 12:46 (`work13/run13b.out`); для REJECT не требуются.
