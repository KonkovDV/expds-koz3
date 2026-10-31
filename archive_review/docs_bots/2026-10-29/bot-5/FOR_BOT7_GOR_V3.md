# Для bot 7: фильтр городищ v3 (patch gate, DINOv2-S) — передача (bot-5, 29.10.2026)

Это измеренная версия на **DINOv2-S**. Версия на DINOv2-B/L-reg пока **не измерена**: kernel готовится для bot 9 (`FOR_BOT9_DINOV2REG.md`), прирост +0,001…+0,004 — оценка из `research/PRETRAINED_WEIGHTS_2026-10-29.md` §4, не наш замер.

## Что даёт (замер, не оценка)
honest-10 к A 0,3146153: **gate Δ +0,005057, LOFO +0,002423**, сиды 5/5 > 0; patch AUC (LOFO) 0,791; 3982 виньетки
(`analysis/gor_patch/gor_patch_eval.json:200,227`; `NOLIDAR.md` §12, таблицы). Вердикт там же: П1–П4 ✓, но меньше MDE, поэтому брать только пакетом.

## Веса
* Бэкбон: `facebook/dinov2-small`, HF revision `ed25f3a31f01632728cabb09d1542f84ab7b0056`, Apache-2.0. `model.safetensors` md5 `ea251be6e151ba8609ad0bc9b2e13a78`
  (`md5sum bot-5/tmp/hf/hub/models--facebook--dinov2-small/snapshots/ed25f…/model.safetensors`).
* Голова: `bot-5/analysis/gor_patch/gor_v3_head.npz`, md5 `acb95c52956d712a16c3dd389237a691`. Ключи: `mu, sd, W(1536×64), qsd, w(65), feat_dim=1536`.
  Обучена на всех 3982 виньетках; чистка Landauer убрала 1603 вхождения (`gor_v3_taus.json`; `gor_v3_fit.log`: 15 с, пик RSS 818 МБ).
* Признаки обучения: `analysis/gor_patch/feats.npz` (md5 `b1720a22f09f76929436948b333a7f5a`) и `meta.json` (md5 `4ce45bd60826a57655f632ed3f5cf01c`).

## Код
Репо `bot-5/repo`, ветка `bot5-gor-patch-kaggle`: abb89d0 (extract_feats, gor_patch_eval, score_cand7) и **82058b8** (`scripts/gor_patch/gor_v3.py`). Не запушено.
1. Признаки новой сцены (только SpOr tif; виньетки 384 м с шагом 192 м и контекст 768 м, 224 px; [CLS, mean] на масштаб, итого 1536-d):
   `DATA=<корень со сценами> OUT=<папка> MODEL=<снапшот dinov2-small> THREADS=2 OMP_NUM_THREADS=2 nice -n 10 python scripts/gor_patch/extract_feats.py`
   На выходе `feats.npz` (F, float16) и `meta.json` (reg, файл, ux/uy — центр в UTM участка).
2. Применение:
   `MODE=apply HEAD=gor_v3_head.npz FEATS=feats.npz META=meta.json CANDS=cands.geojson OUT_GEOJSON=out.geojson OUT_SCORES=scores.json CRS_IN=EPSG:4326 DATA=<корень> python scripts/gor_patch/gor_v3.py`
   Свойства кандидата задаются через `PROP_CONF=conf PROP_REG=reg PROP_MK=mk MK_VALUE=sp_or` (`'*'` — gate для всех). `CRS_IN=utm`, если координаты уже в UTM участка.
3. Переобучение головы (если поменяется набор): `MODE=fit python scripts/gor_patch/gor_v3.py`. Нужны те же feats/meta, что у eval.

**Вход:** GeoJSON кандидатов со свойствами conf, reg (= папка участка), mk.
**Выход:**
* GeoJSON со свойствами `gor_p_scene` (max p по сцене), `gor_p` (ближайшая виньетка того же reg, |dx|,|dy| ≤ 192 м, иначе null) и `conf_gated` (= conf × gor_p_scene для mk == sp_or, иначе conf).
* scores JSON: список `{cand_id, p, p_cov_max, patch, p_scene}`. Это формат `scores_cand7.json` плюс p_scene, без fold.
* **Для оценки** на обучающих сценах берите `analysis/gor_patch/scores_cand7.json`: это OOF без семьи кандидата. Apply на обучающих сценах даёт p в выборке.
  Проверка на 2000 случайных кандидатах bot-7: индекс виньетки совпал с scores_cand7 у 396/396, корр. p 0,764 (in-sample против OOF, так и должно быть). Это `tmp/gor_v3_apply/`.

## Пороги (`analysis/gor_patch/gor_v3_taus.json`)
gate: sp_or **0,992648**; gor_phys 0,43049; li 0,993091; li_coarse 0,986425; li_mask 0,99451. Без gate sp_or был 0,994085, остальные те же.
Как выбирались: `sweep_groups` по обучающим строкам. У sp_or conf умножен на max p сцены из **внутреннего OOF** (5 семейных фолдов, каждая модель с чисткой), то есть ровно та процедура, что внутри каждого фолда `gor_patch_eval.py`. Гиперпараметры (PCA-64, C=1, чистка 0,85/0,05) заданы заранее и не подбирались. Честная цифра процедуры — LOFO выше. Сами финальные пороги обучены на всём обучении, поэтому отдельной оценки у них нет.

## Время и память
| что | замер | файл |
|---|---|---|
| признаки, все 3982 виньетки / 175 tif, CPU 2 потока | 644 с ≈ 0,16 с/виньетку вместе с чтением tif; пик RSS 1467 МБ | `analysis/gor_patch/extract_feats.log` |
| 185_Изборск, 47 виньеток (3 tif) | 6 с вместе с загрузкой модели ≈ 0,13 с/виньетку; RSS 610 МБ | `tmp/gor_v3_time2/extract.log` |
| 046_Тыва_1, 4 виньетки | 1 с; RSS 579 МБ | `tmp/gor_v3_time/extract.log` |
| apply, 2000 кандидатов / 3982 виньетки / 122 сцены | 0,25 с ≈ 0,0001 с/кандидата (RSS не измерен: процесс закончился раньше опроса сторожа) | `tmp/gor_v3_apply/apply.log` |
* На кандидата: почти всё время уходит на признаки сцены. В среднем 32,6 виньетки на сцену (meta.json), то есть ≈ 4–5 с CPU на сцену. Дальше на кандидата ≈ 0,0001 с.
* **T4 — оценка, не замер:** 10–40 мс на виньетку (`research/PRETRAINED_WEIGHTS_2026-10-29.md:77`, [оц]). Скорее всего упрётся в чтение tif: ≈ 1–2 мин на 3982 виньетки.

## Ограничения
12 tif int16/uint16 с неопределёнными каналами: берутся каналы 1–3. Новые точки (add) не помогают (`NOLIDAR.md` §12). Вариант B/L-reg ждёт Kaggle.
