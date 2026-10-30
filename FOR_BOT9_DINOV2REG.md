# FOR_BOT9_DINOV2REG — DINOv2 с регистрами (B/L) для верификатора городищ и кроп-верификатора архитектуры Or

От: bot-5. Ветка `bot5-dinov2reg` (коммит, без push). **Только замер OOF, не сдача, без сабмитов.**

## Что изменено в коде
- `scripts/gor_patch/extract_feats.py`: `AutoModel` вместо `Dinov2Model` — принимает `facebook/dinov2-with-registers-*`;
  регистры не входят в среднее патч-токенов (`num_register_tokens` из конфига); `REVISION` пинует ревизию; на cuda — fp16.
  Для DINOv2-S поведение прежнее (регистров 0).
- `scripts/crop_verify.py`: `--encoder hf:<id>[@ревизия]` — замороженный ViT, признак [CLS, среднее патчей] по 4 отражениям,
  линейная проба (стандартизация по обучению фолда, логрегрессия L2=1e-3 зафиксирована заранее, сбалансированные веса, LBFGS),
  OOF по семьям — как прежняя ветка smp. `--save-model` с hf: отказывает (инференс `expds.crop_model` его пока не читает).

## Запуск
`kaggle/dinov2reg_kaggle.ipynb`, T4 ×1, Internet ON.
- Часть A (городища): smoke `LIMIT_FILES=2` для B и L (assert размерности 4×C), затем полный прогон обоих → `gor_patch_reg.tgz`.
- Часть B (архитектура Or): только если в Inputs есть стек до порогов и эталон трейна (STACK/GT в ячейке); три энкодера:
  `resnet18` (контроль), reg-B, reg-L → `verify_or_reg.tgz`. Нет стека — пропустить, вернуть только A.

Ревизии (HF API, 29.10.2026, все apache-2.0): reg-base `a1d738ccfa7ae170945f210395d99dde8adb1805`, reg-large `e4c89a4e05589de9b3e188688a303d0f3c04d0f3`.

## Оценка T4 (ОЦЕНКА, не замер)
Часть A: на боксе DINOv2-S CPU — см. `analysis/gor_patch/extract_feats.log` (эталон числа виньеток); на T4 узкое место — чтение tif, ViT-B/L на 224 px
по одной виньетке — оценка 15–30 мин на модель. Часть B: зависит от числа кандидатов arkhitektury в стеке; эмбеддинг считается один раз
(не на каждый фолд), проба — секунды; resnet18-контроль — 10 фолдов × 8 эпох, оценка до 1 ч. **Итого оценка 1–2,5 ч T4.**

## Возврат и оценка на боксе bot-5
`tar xzf gor_patch_reg.tgz -C analysis/gor_patch` → `FEATS_TAG=_regB` / `_regL` `python scripts/gor_patch/gor_patch_eval.py` —
те же фолды (fam10, LOFO, seed0-4), сравнение с DINOv2-S и П1–П5. Часть B — табло `scoreboard.py --rerank ... --honest 10`.

## Smoke на боксе (замер, 29.10.2026)
- `extract_feats.py` LIMIT_FILES=1, `facebook/dinov2-with-registers-small`: 015_Демидовка, 15 виньеток, F (15, 1536), всё конечное, 7 с, RSS 590 МБ (`tmp/regsmoke/feats_regS_test.npz`).
- Обратная совместимость: тот же файл с DINOv2-S через новый код — max|Δ| = 0.0 против `analysis/gor_patch/feats.npz` (первые 15 строк).
- `crop_verify.hf_embed` + `lin_fit_predict`, reg-small, 40 синтетических кропов 96 px (диск в центре = «объект»): F (40, 768), 40,6 с на CPU,
  10/10 отложенных разделены верно, RSS 522 МБ. Это проверка кода, не качества.

## Часть B через готовые кропы (29.10, вместо STACK/GT на Kaggle)
- `scripts/crop_verify.py`: три этапа вместо одного прогона (прежний вызов без новых флагов работает как раньше — `run()` = prepare → score → apply_scores).
  - `--dump-crops X.npz` (бокс, CPU): select + labels_of + extract_crops → npz `X (n,3,96,96) uint8, ok, y, y_first, picked, regions, families, fold, meta`.
  - `--from-crops X.npz --p-out P.npz --encoder …` (Kaggle GPU): только oof_verify (hf_embed + lin_fit_predict или smp) → `P.npz (p, picked, ok, y, y_first, meta)`; растры/стек/эталон не нужны.
  - `--apply-p P.npz --stack S --out …` (бокс): проверка `picked` и sha256 стека, затем group_median/blend/отчёт.
- Точность пути (замер, 29.10): подвыборка 3 участка (056_Фалхан, 185_Изборск, 093_Старый_Барах; 72 кандидата, 43 с кропом, 26 проверено), DINOv2-S
  `hf:facebook/dinov2-small`: прежний прогон и dump → from-crops → apply-p дали **байт-в-байт одинаковый geojson** (`cmp`), отчёты совпали (`tmp/cv_test/A.*`, `B*.{log,json}`).
- Датасет для Kaggle (заливает bot 1): **`analysis/or_arch/crops/or_arch_crops.npz`, 14 437 335 B, sha256 `6663649d6f85f4486bb64a478665eb489e7d9eabbe89647dff6de1f28972d44c`**
  (`analysis/or_arch/crops/dump.{log,json}`; 10,5 мин CPU, RSS 518 МБ).
  - Вход: `work/stack_arch_only.geojson` (вся arkhitektury из `/workspace/rerank/stack_arch.geojson` sha b412e299…, потоково, `work/extract_arch_stack.py`; sha256 bcba617f…0a0f),
    GT `/workspace/rerank/gt_train_v2.geojson` sha d4bc1a3b…ff86; фильтр `--markup or --classes arkhitektury --min-conf 0.05 --res 0.5 --tile 96 --folds 10`.
  - 783 кандидата, 626 с кропом, 6 участков = 6 семей. Положительных (y_any) среди кропов **13**: 008_Герите 6/43, 056_Фалхан 7/17; 033_Елизаветинское 0/540, 185_Изборск 0/26.
    014_Лешки (128) и 093_Старый_Барах (29) без кропа: в train-expds-3 у них только разметка и UTM.json, растров нет. Попаданий y_first 20 (из них 10 у участков без кропа).
  - **Мощность замера низкая**: положительные только в двух семьях; OOF фолда Герите учится на 7 положительных Фалхана и наоборот. Результат — подсказка, приёмка только табло honest-10.
- Ноутбук: часть B должна читать npz (`/kaggle/input/*/or_arch_crops.npz`, assert sha), энкодеры resnet18 / DINOv2-S@ed25f3a3 / reg-B / reg-L → `p_or_<tag>.{npz,json}` в `verify_or_reg.tgz`.
  Текст ячеек: `bot-5/tmp/nb_cell{5,6,7}.txt` (см. статус ниже).
- Возврат: `tar xzf verify_or_reg.tgz -C analysis/or_arch/crops` → `crop_verify.py --stack work/stack_arch_only.geojson --apply-p analysis/or_arch/crops/p_or_<tag>.npz --blend geo --out … --report …` → табло honest-10.
