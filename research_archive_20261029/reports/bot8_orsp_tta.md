# bot8 — TTA/мультимасштаб per-layer + compliance submission (LB > 0.41238)

**Цель:** поднять LB выше **0.41238** (build `eb268d6`): per-layer TTA/MS на U-Net (приоритет ae → or → sp_or) при wall ≤20 мин (база 3:48), и compliance только на **submission output** (LESSONS §5).
**Гипотеза:** TTA=4 / MS=[0.75,1.0,1.25] на ae/or (и опц. sp_or) даёт Δ>0 vs v2 mock/LB без просадки слоя >0.01; flags OFF = байт-идентично eb268d6. **Ограничение:** ae на eb268d6 = fort-only (`ae_fort_mask_r18`) τ≈0.998 → R=0.013; TTA не изобретёт курганы (нужны bot3 пороги / D-FINE Ae).
**Критерий приёмки:** Docker `pantum/n7q:20261024` honest-10/mock ALL_F1 Δ>0 к v2 mock **0.3125** / LB **0.41238**; ни один слой не падает >**0.01**; compliance PASS на `result.geojson`; kit-zip → FAIL; `box/run_tests.sh` green.
**Бюджет:** GPU ПК ≤1.5–2 ч (5 вариантов mock + honest-10); один прогон ≤20 мин wall; CPU box ≤0.1 ч тесты.
**Ветка:** `bots/bot8-tta-lb` @ **`f6fe20a`** (база **`eb268d6`**, не 8857343). Worktree: `/workspace/koz3-wt/bot8-tta-lb`. Draft PR: https://github.com/KonkovDV/expds-koz3/pull/9

Обновлено: 29.10.2026 20:35 МСК (Europe/Moscow). D-FINE SpOr — бот 2; D-FINE Ae/Or — бот 3/5 (на eb268d6 слота нет / OFF).

---

## Статус

| Ворота | Результат |
|---|---|
| `box/run_tests.sh bots/bot8-tta-lb` | **PASS** 387 passed, 7 skipped @ `f6fe20a` → `box/results/bots-bot8-tta-lb_f6fe20a.*` |
| compliance kit-guard | unit tests PASS; PC runner points at submission output |
| e2e mock Docker | pending (QUEUE) |
| honest-10 | pending (QUEUE) |
| EXPERIMENTS.csv | только после реальных PC чисел |

## Почему ae recall = 0.013 (LB log)

1. Модель ae = **только фортификации** (`models/ae_fort_mask_r18.pt`, classes=`fortifikatsii`), τ≈**0.998215**.
2. li_to_ae копии режутся `li_ae` / fusion порогами; на ae без лидара копировать нечего.
3. **Нет ae-курганной сети** на публичном LB → сотни GT курганов недостижимы порогами/детектором.
4. D-FINE Ae/Or на дереве eb268d6 **нет** (есть на v4/bot3/bot5, enabled=false до honest-10). Bot2: SpOr D-FINE регрессия/hotfix, не ae.

**Вывод:** TTA/MS на текущем ae U-Net может чуть поднять fort/contour, но **не** ×5 recall. Честный рычаг — bot3 (`ae_recall_relax` / D-FINE Ae) + bot5 Or.

## Сделано @ f6fe20a / cc12f43

1. `Segmenter.ms_scales` + `cfg[tta|ms][markup]` (ae/or/sp_or), default OFF.
2. `scripts/check_compliance.py` + **KIT-ZIP REJECT** (веса / models/ / >100 МБ kit).
3. PC: `run_bot8_compliance.ps1` → `KOZ3_SUBMISSION` = result.geojson; `run_bot8_tta_ms_docker.ps1` → per-layer variants + wall_summary + compliance на out_tta_ms_prio.
4. Draft PR #9.

## Следующий шаг

ПК GPU: Docker gate (baseline/tta_prio/tta_all/ms_prio/tta_ms_prio) → scores vs v2; compliance на submission geojson. EXPERIMENTS.csv — только после цифр.
