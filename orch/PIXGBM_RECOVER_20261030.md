# PIXGBM_RECOVER 30.10 (exec/coord A, 14:30 MSK) — восстановить geo ≥ v2 (LB 0.4124) clean-room, затем + ORG3 + M_rp
Мандат Елены: только рост LB, без регрессий. НЕТ сабмита LB, НЕТ push в main, v3a_min ЗАМОРОЖЕН.

## 1. Корень (подтверждён, LB_DROP_20261030.md)
geo R .656 → .311 (P .769 → .716). v2 = EN-PIXGBM на **всех** методах (maggr finds+poly, etgr, terfl).
v3a_min/v2_clean: `impl="clean"`, **`maps=["etgr"]`** → MagGr (finds_points + object_poly) и TerFl идут **классикой**.
- Полный P-clean (все методы) ранее провалил гейт: **finds −0.15 F1** (PREREG_w4_pix_clean §доп. 23:55) → clean finds для MagGr = главный пробел.
- Ошибка гейта: ORG3 +0.023 мерили на **geo-пуле 18 уч.** против A18, а v2 +0.012 — на **honest-10 всех 103 семей**. Шкалы несопоставимы; v2 на geo-пуле даст намного больше. Гейт надо вести против v2-reference на одном протоколе.
- README_v3a_min «ORG3 alone 0.4124» — ложь (0.4124 = v2 EN). Патч-нота: §6 ниже.

## 2. Инвентарь (14:25)
| Есть | Где |
|---|---|
| clean код | `wt/w4-pix` ветка bots/w4-pix-clean @4da2eb9: expds/pix_clean.py (801 стр.), pix_org.py (ORG1 MagGr), pix_org_et.py (ORG3) |
| clean веса | `w4geo/dist/models_from_v2clean/models/pix_clean/{finds,poly}.txt, meta.json, org/org_etgr_{map,prof}.txt` (finds.txt — та самая модель с −0.15) |
| скрипты | wt/w4-pix/scripts: pix_clean_train.py, pix_clean_oof.py, pix_clean_eval.py, pix_org_oof.py, pix_org_et_oof.py, pix_modality_gate.py |
| GT / OOF A | w5data/gt_train_v2.geojson, w5data/work/table.pkl (honest-10 A 0.314615) |
| TerFl порог | terfl_pix 0.58017 (sweep OOF 18 уч., коммит 26ad1ba/99002ab) — в v2_clean выключен |

| НЕТ (блокеры) | Владелец |
|---|---|
| **geo-растры 200–217 на боксе: 0 tif** (бокс пересоздан; w4geo/eval, w4geo/models_orget, кэши признаков — потеряны) | A: тянуть из S3 train-expds-3 (302 файла, 1.43 ГиБ, только `_MagGr_/_EtGr_/_TerFl_` + разметка + UTM.json). Шарды koz3-w4-geo-s000/s001 **не создавались** (STATUS 29.10 22:30) — искать не нужно |
| v2-reference на geo5/18 уч. (выход EN-PIXGBM как чёрного ящика) | C/coord (не исполнитель clean): прогон v2 eb268d6 на 18 уч. — веса только на ПК Елены / koz3_shared/bot-3 (нет на боксе) → **блокер: Елена** |
| R17 eval json (ORG1/ORG3) | пересчитать после выгрузки данных |

## 3. Гипотеза пути к ≥ v2 (порядок по ожидаемому Δ geo)
1. **MagGr finds clean** (главное): разобрать −0.15 у pix_clean finds (метка r 0.4 м/кольцо, NMS 1.0 м, шаг 0.1 м, honest-порог, калибровка conf между участками) + ORG1-pts как альтернативный/доп. источник (finds +0.042 к C-et, но уч. 200 −0.014). Фикс → OOF geo5.
2. **MagGr poly**: ORG1-poly (A+ORG1poly+ORG3 +0.0194 P_fam .97) vs pix_clean poly.
3. **TerFl pix ON @0.58017** (C-et был etgr+terfl: +0.0187 к A).
4. ORG3 (EtGr) — уже есть. M_rp — поверх.
Кандидат C* = pix_clean(maggr-fix, etgr, terfl) | ORG1(лучшее из pts/poly) + ORG3 + M_rp.

## 4. Гейт (обязателен до «Elena-ready»)
- Один протокол: OOF geo5 (F1={200} F2={201..207} F3={208,209,210,215} F4={211,212,216} F5={213,214,217}), 18 уч., gt_train_v2, честные пороги.
- Сравнение **C* vs v2-ref** и vs A18; требование: geo P ≥ v2, R ≥ v2 (на LB это .769/.656), ΔF1 finds и poly ≥ v2-ref, ни один метод < −0.003.
- Если v2-ref недоступен: суррогат = **v2 должен быть ≥ +0.012 honest-10** → требуем C* honest-10 Δ vs A ≥ +0.012 (п. §8 spec: ≥ +0.009) **и** geo-pool Δ ≥ geo-pool v2 (неизвестно → не Elena-ready, только «кандидат»).

## 5. Раздача
- **A (coord/exec, я)**: (1) ✅ geo 200–217 на боксе: **канон `/workspace/koz3/w4geo/train/`** (C, 320 файлов 1.5G; мой дубль raw/ удалён, побайтно тот же список) [нужна box-аренда: box/cpu сейчас у bot-W2 до 19:35 — I/O лёгкое, не тяжёлый процесс]; (2) перестроить кэш признаков pix_clean; (3) обучение/OOF finds-fix на Kaggle CPU (аренда acc2/cpu после 17:14 или свободный пул) или боксе (≤3 проц, ≤8 ГБ, nice 10). Kernel push только с арендой.
- **C (W4 gate, агент 3)**: харнесс `pix_modality_gate.py` → режим «C* vs REF» (REF = любой geojson на 18 уч.), сводка P/R geo по методам и классам, формат как LB scoring.log; TerFl pix @0.58017 в пакетный конфиг (флаг, деф. выкл. до гейта); **v3a_min заморожен**, без LB.
- **B**: (если свободен) ORG1-pts фикс для уч. 200 (Судьбищи −0.014) — диагноз по выходу после выгрузки данных.
- **D**: CPU-слоты после seeds-b → отдать под OOF geo5 (5 фолдов × finds/poly).
- **Елена (блокер)**: v2-ref — либо прогон v2 eb268d6 на 18 гео-уч. на ПК (`F:\expds3\work` eval_geo), либо выгрузка его geojson; решение «да» на любой LB.

## 6. Патч-нота к README_v3a_min (не переписываю историю)
`w4geo/dist/README_v3a_min.PATCH_20261030.md`: «Prior LB probe ORG3 alone: 0.4124» — НЕВЕРНО. 0.41238 = bot7cand_v2 eb268d6 (EN-PIXGBM, все методы). ORG3 отдельно на LB не мерился. v3a_min на LB = 0.37838.

## 7. Лог
- 14:30 бриф записан. Дальше: выгрузка geo 200–217, ETA данных ~15–30 мин; кэш + OOF finds-fix ~3–5 ч CPU.
- 14:32 канон данных = w4geo/train/. C: GO харнесс + флаг TerFl. A/exec: MagGr finds-fix + кэш признаков + пересборка OOF. v2-ref не блокирует (запрос Елене в работе).
- 14:38 venv `/workspace/koz3/w4geo/venv` (pin из requirements.lock: numpy 2.5.4, scipy 1.18.1, shapely 2.2.0, pyproj 3.8.0, rasterio 1.5.2, lightgbm 4.7.0). Классика A18 → `w4geo/work/cache` (pix_clean_oof.py classic, configs/v3_on.toml).
- 14:45 **A18 воспроизведён побайтно**: `w4geo/work/A18.geojson` sha 02281f62… = гейт 01:25; score2 0.42266, F1 finds 0.5536 / poly 0.2917 (eval `w4geo/work/R_AvsA.json`, EXPDS_METRIC=materials/compute_metrics.py; config.py метрики нет → дефолты, как и раньше).
- 14:46 Драйвер `w4geo/work/run_finds_fix.sh` (nohup, nice 10, ≤3 потока; лог `w4geo/work/driver.log`): sample признаков (97 гео-растров, `work/feat`) → fit 5 фолдов s0 (`work/models/F*_s0`) → pix OOF d10 (все методы) → compose 8 вариантов → `work/R_d10_vsA18.json`.
  Варианты: Sall (полный P-clean, ожид. finds≈−0.15 — репро), Cet, Setonly, Umag (union MagGr, dedupe ≤1 м), SmagPoly, SmagFinds, UmagFinds, CetUmagFinds.
  ETA: sample ~15:20, fit ~15:40, pix ~16:40, eval ~17:00 MSK.
- Дальше (после d10): sweep декодирования finds на OOF (nms 0.5/1.5, p_min, conf-калибровка), ORG1-pts/poly OOF (pix_org_oof.py), ORG3 OOF (pix_org_et_oof.py) → кандидат C* = лучший MagGr + Cet + ORG3; гейт C (харнесс) vs A18 и vs v2-ref, когда Елена даст.
- 14:47 (exec) **v2-ref блокер снят**: Елена прислала zip → `w4geo/ref/submission_bot7cand_v2_eb268d6.zip` (501M, sha 8f3f2671…). DOC-1: EN-код (geo_pixgbm.py и т.п.) НЕ открывать; только black-box прогон на 18 гео-уч. → REF geojson. Владелец REF-прогона: **C** (exec держит CPU под finds-fix/ORG); C: прогон в изолированном каталоге, наружу только geojson.
- 14:47 (exec) статус: fit F1–F5 ✓ (14:42); pix d10 идёт. Параллельно `w4geo/work/run_org.sh` (nice 10, 2 потока): ORG3 cand ✓ → ORG3 fold → ORG1 cand → ORG1 fold → `cache/orget/s0`, `cache/org/s0`. Готов sweep декодирования finds `w4geo/work/sweep/finds_sweep.py` (dump карт вер. по фолд-моделям → σ{1,2,3}×NMS{0.5..2.0}×p_min×калибровка{raw,q999,max} → кэш-теги для compose).
- 15:40 (exec) **d10 итоги** (`w4geo/work/R_d10_vsA18.json`, Δ = d_global vs A18): Sall +0.0030 (finds −0.1425 — репро провала), Cet **+0.0187**, Setonly +0.0160, SmagFinds −0.0142, UmagFinds −0.0062. ORG (`w4geo/work/c2/R_org.json`): **CetO3 +0.0225** (=CtO3: ORG3 заменяет etgr везде, где есть), CetO3O1poly +0.0204 (5/5 фолдов, p 1.00), CetO3O1p +0.0212, CetO1 +0.0155 → ORG1 finds −0.013 F1, ORG1 poly −0.002 — ORG1 не берём.
- 15:40 (exec) C-гейт vs REF_v2: C_Cet FAIL, MagGr R .508 vs .859. NB: REF = v2 на трейн-участках (модель EN видела эти 18 уч. → in-sample), C = OOF; разрыв R частично артефакт протокола. Тем не менее finds — главный пробел.
- 15:40 (exec) **Диагноз finds**: декодирование не лечит (sweep 51 вариант σ/NMS/p_min/калибровка, `c3/eval.log`: лучший fs1_n1_max −0.0127 F1). Pix Rmax 0.88 vs классика 0.59 — кандидаты есть, ранжирование плохое (уч. 200 oracle F1 0.15 vs кл. 0.63). Стекер кандидатов (`sweep/finds_stack.py`) хуже. **Корень — недовыборка фона в обучении** (NEG_RATIO 3: ~33k фоновых пикс на 14 Мпикс растр).
- 15:40 (exec) **Ретрейн finds NEG_RATIO 15 / NEG_MIN 50k** (`sweep/retrain.py`, только константы, код pix_clean_train): впервые finds > A18: S n15_fs1_n1_max **finds +0.032**, union-с-классикой U n15_fs1_n0.75_raw **finds +0.053 (Δ +0.0053)** (`c4/R_n15.json`). Дальше: NEG 40/150k (идёт, `sweep/pipe.sh n40`), pos_r 0.6; затем сборка C* = лучший MagGr finds + Cet/ORG3 (TerFl OFF по C).
- 16:15 (exec2) **NB баг pipe.sh**: compose/eval шли с `../venv` после `cd wt/w4-pix` → n40 compose/eval молча падали. Исправлено (абс. путь в pipe.sh; `sweep/ce.sh NAME` = хвост compose+eval). n40, n40r6 досчитаны.
- 16:15 (exec2) **finds-ретрейн итоги** (Δ vs A18, только MagGr finds, pix_clean_eval): n15 U_n0.75_raw finds +0.053 (Δ+0.0053); n40 S_n1_raw finds +0.035 (Δ+0.0035) — NEG сверх 15 не помогает; **n40r6 (pos_r 0.6) S_n1_max finds +0.073 (Δ +0.0073, 3/5 фолдов, p .86)**, S_n1_raw +0.063 (p .97) — лучший. `w4geo/work/c_n40r6/R.json`.
- 16:15 (exec2) **C\* собраны** (`w4geo/work/cstar/`, `sweep/merge_finds.py`: CetO3 + MagGr finds из варианта; остальное байт-в-байт CetO3). Гейт `pix_geo_pr_gate.py` vs REF_v2 (honest Δ vs A = global, raw-файл; P/R — после честных порогов по фолдам `_ht`, `sweep/honest_filter.py`; raw P/R без порогов несопоставим с REF):
  | C\* | honest Δ vs A | geo P/R (_ht) | MagGr P/R (_ht) |
  |---|---|---|---|
  | CetO3 (было) | +0.0244* | .612/.488 | .669/.460 |
  | CetO3_Un15 | +0.0278 | .548/.628 | .567/.631 |
  | CetO3_Sn40 | +0.0260 | .492/.688 | .495/.704 |
  | **CetO3_Sr6max** | **+0.0298** | .554/.658 | .574/.668 |
  | CetO3_Sr6raw | +0.0288 | .532/.674 | .544/.687 |
  | REF_v2 (in-sample) | — | .664/.785 | .671/.859 |
  (*CetO3 через _ht; pix_clean_eval давал +0.0225.) Гейт vs REF всё ещё **FAIL** (P и MagGr R ниже REF), суррогат ≥+0.012 PASS.
- 16:15 (exec2) **Остаток дыры = Гнёздово**: по сайтам (Sn40/Sr6 _ht) Семибратнее 201–207 почти = REF (201 .83 vs .94, 202–206 ≥.94), а 209 .14–.22 vs .73, 211 .08–.12 vs .59, 210 .42–.50 vs 1.0, 212 .42–.50 vs 1.0, 200 .64–.74 vs .94. pix-карты на Гнёздово дают мало кандидатов выше общего порога → калибровка/домен (F3/F4 видят в обучении почти только Семибратнее/Судьбищи). Следующая итерация: n40r8 (pos_r 0.8) запущена 16:14 (`sweep/run_iter.sh`), ETA ~16:45; дальше — калибровка по растру для Гнёздово / union с классикой только там, где pix пуст.
- 17:25 (exec3) **n40r8/n40r10 итоги + Гнёздово-фикс**. Бокс перезагрузился ~17:06 (restore 17:07–17:12): `w4geo/venv` и `w4geo/ref/out/` (REF_v2.geojson) НЕ восстановлены, docker на боксе отсутствует. venv пересобран из requirements.lock (numpy 2.5.4/scipy 1.18.1/shapely 2.2.0/pyproj 3.8.0/rasterio 1.5.2/lightgbm 4.7.0) — паритет ✓ (C_CetO3_Sr8raw P/R .286009/.789520 и honest +0.0317 байт-в-байт как до ребута). n40r10 pipe до ребута дошёл до eval (17:03), R цел.
  - pos_r: n40r8 (0.8) S_n1_raw finds +0.092 (Δ +0.0092, p 1.00) — **оптимум**; n40r10 (1.0) S_n1_raw finds +0.059 (Δ +0.0059) — хуже. `c_n40r8/R.json`, `c_n40r10/R.json`.
  - Per-raster калибровка conf (`sweep/calib_union.py`, без GT): pct/q99/blend — все хуже базы (+0.0028…+0.0080 vs +0.0092). Union с классикой где pix пуст (d 1–3 м, p_min 0/0.5, своя threshold_group) — 0…−0.001. `c5/R.json`. Не берём.
  - **Threshold_group по тест-времени признакам растра** (`sweep/raster_groups.py`: тип карты серый/цвет × число pix-кандидатов на растр < cut): honest-пороги по фолдам теперь свои на группу. Gtype_n300 finds +0.116 (Δ maggr +0.0116); робастность по cut: n150 +0.0106, n600 +0.0098 (все > база +0.0092); одиночные группировки (q90/med/nhi/n/type) хуже. NB: выбран из ~17 вариантов → лёгкий selection bias; на инференсе нужны taus на группу `maggr_pix_{gr,cv}{lo,hi}` с полного трейна. `c5/Rg.json`, `c6/R.json`.
  - **C\* = `cstar/C_CetO3_Sr8G.geojson`** (sha 5d815cb4…; CetO3 + MagGr finds из `c5/C_r8g_Gtype_n2.geojson`; _ht sha b708e338…): honest Δ vs A18 **+0.0341** (5/5 фолдов, p 1.00; было Sr6max +0.0298, Sr8raw +0.0317), finds F1 .669. P/R (_ht): geo .604/.669, MagGr .638/.681 (Sr8raw_ht .576/.661, .602/.671). REF_v2 (до ребута) .664/.785, MagGr .671/.859 → гейт vs REF всё ещё **FAIL**, суррогат PASS. `cstar/R_cstar_1720.json`, `cstar/gate_CetO3_Sr8G{,_ht}.log` (сравнение с REF упало: REF файла нет).
  - По сайтам MagGr R (_ht, Sr8raw→Sr8G): 209 .23→.61, 208 .46→.85, 211 .12→.23, 210 .59→.58, 212 .56→.36 (↓), Семибратнее −0.02…−0.05. 211 — у pix мало кандидатов даже raw (R .28) → нужен домен/данные, не порог.
  - Блокер re-gate: REF_v2.geojson пропал при ребуте; перегенерация = `w4geo/ref/run_ref_emit.sh` (владелец C, нужен docker + образ pantum/n7q:20261024 — на боксе после ребута docker нет).
  - NB: slice terfl +0.0142 vs A18 присутствует во всех CetO3-C* (унаследовано от Cet, не pix TerFl-флаг) — C, проверить, что это не TerFl pix ON.
- 17:50 (exec4) **C\* без TerFl pix (баг C подтверждён и снят)**. Cet-линия (`run_finds_fix.sh` comp Cet, maps=[etgr,terfl]) несла TerFl pix: 252× `threshold_group=terfl_pix`. Пересборка: `pix_clean_oof.py compose --tag d10 --orget-tag s0 replace=true maps=["etgr"]` → `cstar/noterfl/C_SetO3.geojson` (контроль: тот же compose с maps=[etgr,terfl] = c2/C_CetO3 побайтно, sha a831d66f; merge c2/CetO3 + c5/Gtype_n2 = C_CetO3_Sr8G побайтно, sha 5d815cb4) → `merge_finds.py` с тем же MagGr finds Sr8G (`c5/C_r8g_Gtype_n2`) → **`cstar/C_CetO3_Sr8G_noTerFl.geojson`** (21647 объектов) + `_ht` (7026). **terfl_pix = 0**; TerFl = 319 классических, побайтно как в A18.
  - honest Δ vs A18 (pix_clean_eval, `cstar/R_noTerFl.json`): **+0.0322** (было +0.0341 с TerFl pix → TerFl pix давал лишь **−0.0019** глобально, object_poly ΔF1 .2250→.2060; не 0.014 — «slice +0.0142» C был срезом гейт-харнесса, не глобальным вкладом). 5/5 фолдов, p_fam ≥.95, accepted. SetO3 без MagGr finds: +0.0206.
  - Гейт vs REF_v2 (exec pre-check, `cstar/gate_CetO3_Sr8G_noTerFl{,_ht}.json`): _ht geo P/R **.602/.663** (было .604/.669), MagGr .638/.681 (без изм.), EtGr .472/.682, TerFl .072/.034 (класс., REF .000/.000); raw geo .284/.786, MagGr R .819. **FAIL** vs REF (P и MagGr R); суррогат PASS (+0.035 _ht / +0.032 raw в харнессе).
  - Опц. «смягчить MagGr honest thr» (`sweep/honest_filter_soft.py`, tau×MUL для maggr_pix_*; диагностика, `cstar/soft/`): ×0.75 MagGr P/R .575/.732 (F1 .644), ×0.5 .499/.776 (F1 .608) vs база .638/.681 (F1 .659). R растёт, F1 падает, до .859 не дотягивает → **не берём**; дыра MagGr R — ранжирование/домен Гнёздово (211/212), не порог.
  - Кандидат на re-gate C: `cstar/C_CetO3_Sr8G_noTerFl{,_ht}.geojson`. v3a frozen, no LB, DOC-1 (EN не открывался).

- 18:35 (exec-gnz) **Гнёздово MagGr recall: детектор замкнутых изолиний + C\*_noTerFl_L.** Диагностика: 211 — на цвет_2 (0.022 м) находки = мелкие замкнутые петли изолиний, на шаге 0.1 м размазываются; на 211 серый/цвет prob AUC@1м ≈ .43–.47 (нет hp-аномалии; лейбл-мисматч 95 vs 314 точек). 212 — AUC .875, но ранжирование/порог (cvhi tau .40), GT сдвиг ~0.9 м. 209/210 — натив 0.183 м, полосы съёмки, палитра ломается PCA/luma.
  - Детектор петель (без GT при детекции; параметры подобраны глядя на 211/212 → лёгкий in-sample bias): `w4geo/work/gnz/{loops,emit_loops,add_loops}.py`, `gnz/loops.json` (325 петель, только 211/212 цвет_2; на других картах 0). Union с базой: +255 точек (211→cvlo, 212→cvhi), 56 boost.
  - **`cstar/C_CetO3_Sr8G_noTerFl_L.geojson`** (sha 8b7445c2, 21902; _ht sha eb30e9eb, 7296), **terfl_pix=0**. Δ vs A18 **+0.0351** (база noTerFl +0.0322), finds F1 .699 (+0.145), 5/5, p 1.00. Гейт _ht: geo .611/.704 (было .602/.663), MagGr **.648/.732** (было .638/.681), REF MagGr R .859 → **FAIL** (P<REF, MagGr R −0.127). Raw: MagGr R .862 ≥ REF. Сайт R (_ht): 211 .232→.631, 212 .359→.705, остальные без изменений.
  - Отклонено: веса Гнёздово в обучении (gw=3/1, drop 211 серый/цвет) — finds-only +0.0033/+0.0012 vs база +0.0092; апсемпл 209/210 до 0.1 м на тесте — finds-only +0.0101 (vs +0.0092), но в C\* с Gtype_n300 +0.0325 < L +0.0351.
  - Для LB: петли надо встроить в inference-пакет (pix decode hook). LB не сабмитили.

- 19:20 (exec-m910) **209/210 MagGr recall + P-lift итерация после C\*_noTerFl_L.** Работа: `w4geo/work/m910/` (DOC-1: EN не открывался; без push/LB).
  - Диагноз 209/210 (L, raw): нет сист. сдвига pix↔GT (медианы dx/dy <0.5 м, сдвиг не помогает); покрытие GT кандидатом ≤2 м: 209 .52, 210 .65 (A18 классика .30/.54). 209 — дефицит кандидатов (много находок без видимой аномалии на карте), 210 — порог (raw .96 → _ht .58, цвет в cvhi tau ~.4–.5).
  - **Res-аугментация обучения** (`m910/sample_aug.py`: каждый MagGr растр <0.15 м дополнительно деградирован до 0.1833 м average-WarpedVRT, константы n40r8, регионы сохранены → фолды честные; `feat_aug`, `models_aug`, `c_aug`): finds-only S_n1_raw +0.0077 (база n40r8 +0.0092). Роутинг по тест-признаку res>0.15 → aug-модель (`m910/route.py`, chain Gtype_n300+merge+loops `m910/chain.sh`, воспроизводит L побайтно): кандидаты 209 .52→.60, 210 .65→.74 (лучше локализация), но honest Δ: Ar +0.0326, n150 +0.0329, n600 +0.0341, n1000 +0.0329 (сдвиг групп 209/210 в cvhi), с группами L (`ArgL`) +0.0363 vs без aug +0.0365. Гейт _ht LAug_noTFc: 209 R .614→.674, 210 .579→.528, MagGr R .730 (≈L). **Отклонено** (не бьёт L-линию по honest).
  - Union классики на грубых растрах (res>0.15, dedupe 1/2 м, группа pix-растра) +0.0347/+0.0348 < L .0351. **Отклонено**.
  - **P-lift: убрать классический TerFl** (P_ht .072, ~155 FP / 6 TP; REF TerFl = 0 предсказаний): honest Δ **+0.0365** (L +0.0351; object_poly ΔF1 .2060→.2195), 5/5 фолдов, accepted. terfl_pix=0, TerFl-предсказаний 0.
  - **C\* = `cstar/C_CetO3_Sr8G_L_noTFc.geojson`** (sha 4eb6726f…, 21583: maggr 20005, etgr 1578) + `_ht` (sha 77f0bb18…, 7185: maggr 5682, etgr 1503). Гейт (`cstar/gate_CetO3_Sr8G_L_noTFc{,_ht}.json`):
    | | geo P/R | MagGr P/R | EtGr P/R | honest (харнесс) |
    |---|---|---|---|---|
    | L raw | .29/.82* | .277/.862 | — | +0.0380 |
    | L _ht | .611/.704 | .648/.732 | .472/.682 | +0.0380 |
    | **noTFc raw** | .294/.816 | .277/.862 | .462/.720 | +0.0365 |
    | **noTFc _ht** | **.618/.704** | .648/.732 | .463/.688 | +0.0382 |
    | REF_v2 | .664/.785 | .671/.859 | .571/.517 | — |
    (*L raw geo по C 19:10.) Гейт vs REF: **FAIL** (geo P и MagGr R _ht ниже REF; raw P ≪ REF). Суррогат PASS. Остаток P-дыры: EtGr ORG3 ~1000 FP (REF ~510) и MagGr finds R (REF in-sample). Elena_ready: **нет**; кандидат на re-gate C — да (строго лучше L по honest и geo P_ht).

- 20:10 (exec-etfp) **EtGr/ORG3 FP cut на L.** Работа: `w4geo/work/etfp/` (DOC-1: EN geo_pixgbm не открывался; без push/LB; terfl_pix=0; MagGr finds L побайтно без изменений — проверено, меняется только etgr).
  - Диагноз (L, matching как в метрике): EtGr raw 1578 = 474 TP / 1104 FP; _ht 1453 = 449/1004. FP сидят в 214 (403 _ht FP / 126 TP; 529 предсказаний на 177 GT) и 215 (284/95). На растр ~20–35 полигонов при ~5–15 GT. Ранг в растре (AUC .74) информативнее conf (.72); крошечные полигоны (<0.2 м²) TP-rate .15. Honest tau etgr_org: .216 (F1–F4), **0.0 для F5** (213/214/217 без фильтра).
  - **Правило (без GT): на каждый EtGr-растр top-K по conf + площадь ≥0.2 м²** (`etfp/mk_E.py K A NAME`). Плато K 22–30 × a .15–.3: honest Δ +0.0362…+0.0373 (все ≥ L).
  - **C\* E1** = `cstar/C_CetO3_Sr8G_noTerFl_L_E1.geojson` (K=25, a=.2; sha 963ec28e, 21515: maggr 20005, etgr **1191**, terfl 319) + `_ht` (sha fb496315, 7025: maggr 5682, etgr 1182, terfl 161).
  - **C\* E2** (high-P) = `cstar/C_CetO3_Sr8G_noTerFl_L_E2.geojson` (K=15, a=.2; sha 68a512e9, 21207: etgr **883**) + `_ht` (sha ee2463a7, 6726: etgr 883).
  - Гейт vs REF_v2 (`cstar/gate_CetO3_Sr8G_noTerFl_L_E*{,_ht}.json`) + honest pix_clean_eval (`cstar/R_E.json`):
    | | geo P/R | MagGr P/R | EtGr P/R (n) | honest Δ (pce) | proxy |
    |---|---|---|---|---|---|
    | L raw | .292/.820 | .277/.862 | .462/.720 (1578) | +0.0351 | accepted |
    | L _ht | .611/.704 | .648/.732 | .472/.682 (1453) | | |
    | **E1 raw** | .294/.812 | .277/.862 | .539/.667 (1191) | **+0.0373** | accepted 5/5, p 1.0 |
    | **E1 _ht** | **.626/.702** | .648/.732 | .539/.663 (1182) | | |
    | E2 raw | .293/.798 | .277/.862 | .600/.575 (883) | +0.0356 | accepted 5/5 |
    | E2 _ht | .636/.688 | .648/.732 | .600/.575 (883) | | |
    | E1_noTFc _ht | .636/.702 | .648/.732 | .538/.666 | +0.0394 | **NOT accepted** (terfl slice −0.033 → methods_ge_-0.003 false) |
    | REF_v2 | .664/.785 | .671/.859 | .571/.517 | | |
  - Отклонено: OOF-LGBM ресскоринг etgr (LOFO F3/F4/F5, 10–19 GT-free фич; AUC .739 vs conf .718) — +0.0351…+0.0361, P_ht ≤.514; split threshold_group map/prof +0.0353 P .502; cap≤12 (honest < L); OOF-ранг cap18+a.2 +0.0367 P .595 R .602 (сложнее E1/E2 без выигрыша). Классический TerFl **оставлен**: drop даёт P_ht +.009 и Δ +0.002, но валит methods_ge_-0.003 (terfl −0.033) — R_noTF.json тоже accepted=false (запись 19:20 «accepted» неверна).
  - Гейт vs REF: **FAIL** (c_ge_ref_p и c_ge_ref_r). Арифметика: E1 _ht geo tp 3202 / fp 3823 → для P .664 при том же TP нужно fp ≤3234 (−589), а у EtGr _ht осталось ~746 FP всего; geo R .702 vs .785 (−376 TP) — это MagGr recall. Только EtGr гейт не закрыть. K/a выбраны по тем же фолдам (лёгкий in-sample bias; плато широкое). Elena_ready: **нет**. Re-gate C: да (E1 основной, E2 high-P). No LB.

- 20:25 (exec-magp) **MagGr _ht P/R lift on E1 (finds-only, GT-free test-time rules).** Work: `w4geo/work/magp/` (DOC-1: EN geo_pixgbm not opened; no push/LB; terfl_pix=0; etgr/terfl/maggr-poly byte-identical to E1, only the `confidence` of maggr finds_points changes, n is the same as E1 = 21515).
  - Diagnosis (E1, matching as in the metric; `magp/rk.py`, `src.py`, `dup.py`): raw finds R .902 → _ht .763. At equal n per site REF (in-sample) has much higher R (200: .94 vs .66; 207: .84 vs .79; 210: 1.06 vs .74; 212: 1.0 vs .73) → the gap is ranking within the raster, not a threshold issue. In _ht there are 1382 same-raster pairs ≤2 m, and in 67% of them the lower one is FP (duplicates/halo of an anomaly) → P lever. 207 conf scale is low (hit median .31 vs ~.5 elsewhere, grhi tau .27) → per-raster normalisation lever.
  - Fast screen `magp/ev.py` (honest finds F1 per geo5 folds + MagGr P/R; matches the gate to ±.002).
  - **Rules (`magp/mk_M.py NAME Q A D X`)**: (1) per-raster norm c' = 0.9·c^A·(c/q_Q(raster))^(1−A); (2) soft-NMS: greedy by c', lower MagGr finds ≤D m (same region, any MagGr raster) c'×=X. Groups are as in L (Gtype_n300). Plateau grid q{.85,.9,.95}×A{.25..1}×D{1.75,2.0}×X{.6..8}: all 63 points ≥ E1 by finds F1 (.703–.713 vs .699); **D 2.25 is a cliff** (honest tau jump, R .62–.70) → D ≤2.0. Parameters were chosen on the same folds (light selection bias, like K/a for E1).
  - C\* (`w4geo/work/cstar/`):
    | | geo P/R _ht | MagGr P/R _ht | finds P/R _ht | honest Δ pce | harness Δ (_ht) | accepted |
    |---|---|---|---|---|---|---|
    | L | .611/.704 | .648/.732 | | +0.0351 | +0.0380 | yes |
    | E1 | .626/.702 | .648/.732 | .648/.763 | +0.0373 | +0.0393 | yes |
    | E1_M1 (soft D2 X.7) | .647/.693 | **.675**/.720 | .675/.751 | +0.0384 | +0.0404 | yes 5/5 |
    | E1_M2 (q.9 A.5 + soft D1.75 X.6) | .642/**.704** | .669/**.735** | .669/.766 | +0.0387 | +0.0405 | yes 5/5 |
    | **E1_M3** (q.9 A.5 + soft D2 X.7) | **.644**/.702 | **.671**/.732 | .672/.764 | **+0.0387** | +0.0406 | yes 5/5 |
    | REF_v2 | .664/.785 | .671/.859 | .674/.894 | | | |
    sha: M1 5766ae82 (_ht 976047632), M2 2af41327 (_ht 9a4cf389), M3 f71929b3 (_ht f44b7473). Gate JSON `cstar/gate_CetO3_Sr8G_noTerFl_L_E1_M{1,2,3}{,_ht}.json`, honest `cstar/R_M.json`.
  - M3: MagGr P_ht .6713 ≥ REF .6705 (closed, but by +.0008, within noise); MagGr R .732 (= E1, −.127 vs REF). M2: R +.003, P .669. Sites (_ht R, E1→M3): 207 .53→.59, 202 .91→.94; 212 .71→.62, 210 .58→.53, 205 .89→.87.
  - Rejected: (a) OOF/nested LGBM rescoring of MagGr candidates (30 GT-free features: raster stats, rank, neighbours/cross-raster, area; also with init_score=logit(conf) and monotone conf) — OOF AUC .81–.87 < conf .88, worse on 3/4 folds (domain shift between folds), honest R .29; (b) hard NMS 1.25–3 m (R drops more than P rises, F1 ≤ E1); (c) regrouping 210 colour → cvlo (n500/n1000/res>.15): 210 R .57→.96, but 212 .71→.18 (cvhi on F4-held is left with 200 only) → F1 .681–.693 < E1; cv-single/type-only/cv2 also worse; (d) union with classic A18 MagGr finds >1/2 m from pix: 273 new, of which 34 hit → F1 .599–.708 ≤ soft-only; (e) per-raster pct-rank (F1 .673).
  - Conclusion: post-processing closes the **P** axis of MagGr (and geo P .626→.644), but not **R**: R_ht is limited by ranking within the raster plus fold thresholds with 1–3 rasters per group; REF R .859 is in-sample. Geo P .644 < REF .664, geo R .702 < .785 → gate vs REF **FAIL** (c_ge_ref_p, c_ge_ref_r, maggr ΔR −.126). Elena_ready: **no**. For C to re-gate: **E1_M3** (main), M2 (max R), M1 (max P). For inference: soft-NMS + per-raster norm are pure test-time ops; group taus need to be recomputed on the new conf scale from the full train set.

- 21:55 (exec-night, MSK) **MagGr R grind — interim.** Work: `w4geo/work/night/` (DOC-1: EN geo_pixgbm not opened; no push/LB; terfl_pix=0; classic TerFl 319 kept).
  - Diagnosis (E1_M3 _ht vs REF, `magp/rkM3.py`, `night/opt_cut.py`): TP gap ≈ 470. Per site at REF's own count n (oracle per-site cut) ours vs REF TP: 2810 vs 2910 → ranking explains only ~100; rest is the cut. Biggest holes: 207 (−106 TP, kept 346 vs REF 613), 210 (−84), 200 (−63, real ranking deficit: @REFn R .65 vs .94), 201 (−41), 203/205 (~−40), 209 (−33), 212 (−30). Group taus are stable over folds (grhi ≈.32), but per-raster optimal cuts vary .19 (201/207) – .46 (206); no GT-free raster statistic predicts it (q90, density, n>.5 checked).
  - **In-sample check**: sweep_groups on ALL folds (non-honest taus) for E1_M3 → finds P/R .679/.763, MagGr .679/.731 — same R as honest. ⇒ the −.127 MagGr R gap is NOT honest-vs-in-sample thresholds; REF_v2 conf is from a model fitted on these rasters (memorised ranking). With 4 threshold groups and OOF conf, R ≈ .73 is the F1-optimal operating point.
  - Map-feature rescoring at E1 points (σ2/σ4, disc mass, ring/prominence, half-max area; `night/mapfeat.py`, `rx_g1.py`): all ≤ E1_M3 (conf AUC .875 is best single). Rejected.
  - **OOF prob-map ensemble** (mean of n40r8, n40r6, n40r10, aug, gw1d, gw3d — all same geo5 folds, OOF): within-raster AUC .875→.881. Rescoring E1 candidates (R1–R3) or decoding the mean map (D1/D2, `night/pipe_ens.sh`, probs_ens6) raises finds F1 .713→.720–.723 and MagGr/geo P, but R stays ≤.73 (F1-opt point moves toward P).
    | | geo P/R _ht | MagGr P/R _ht | honest Δ pce | accepted |
    |---|---|---|---|---|
    | E1_M3 | .644/.702 | .671/.732 | +0.0387 | yes |
    | E1_R1 (ens6 rescore, A.25 D1.75 X.7) | .659/.697 | .691/.725 | +0.0395 | yes 5/5 |
    | E1_R3 (A.25 D2 X.7) | **.668**/.686 | .703/.712 | +0.0395 | yes 5/5 |
    Running: full-train in-sample model (like REF) for an apples-to-apples check; bigger LGBM (n40r8L, 127 leaves/800 rounds); gates for D1/D2.

- 22:55 (exec-night, MSK) **MagGr R grind — final. Wall confirmed structural; best C\* = E1_G1.** Work `w4geo/work/night/` (DOC-1: EN geo_pixgbm not opened; no push/LB; terfl_pix=0 in all; classic TerFl 319 kept; etgr/terfl byte-identical to E1).
  - Orphans adopted (not killed): `sweep/pipe.sh n40r8L` (127 leaves/800 rds) ran to completion; `pipe_ens.sh ens6m` (probs_ens6 mean map, grid main, raw) — 17 tags chained+merged to E1 (`night/E1_ens6m_*`). First ens6m screen failed (EXPDS_METRIC unset) → rerun OK.
  - **n40r8L (bigger LGBM): rejected.** finds-only honest Δ best +0.0048 (S n1 max) vs n40r8 +0.0092 — more capacity = more overfit to train rasters (`c_n40r8L/R.json`).
  - **Full-train in-sample model (apples-to-apples with REF)**: `retrain.py fit` on all 13 MagGr regions → `models_n40r8full/ALL_s0`, used for every region (`models_n40r8ins` symlinks), same decode/compose/chain/E1 merge (`night/pipe_ins.sh`) → `cstar/C_..._E1_INS{,_ht}` (diagnostic ONLY, not for C/LB). Global MagGr-finds PR curve (`night/prcurve.py`, metric P=tp/(tp+.5fp)):
    | finds P at R= | .73 | .80 | .859 | F1-opt |
    |---|---|---|---|---|
    | REF_v2 | .820 | .786 | .736 | .797 (P.770/R.825) |
    | **ours in-sample (INS)** | .906 | .856 | **.754** | .828 (P.853/R.804) |
    | ours OOF E1_M3 | .634 | .544 | .420 | .680 |
    | ours OOF D1 (ens6) | .638 | .552 | .444 | .686 |
    ⇒ our model fitted in-sample **dominates REF** at every recall; OOF at R .859 gives P ≈.42–.44 (< .671 needed). The −.127 MagGr R gap = in-sample vs out-of-fold ranking (REF memorised train rasters), not thresholds/post-processing/capacity. INS gate _ht: MagGr .858/.720, geo .782/.693, Δ +0.0477 (in-sample — meaningless as honest). **Wall structural: confirmed.** Honest MagGr R_ht ≥.859 at P ≥.671 is unreachable with this data/feature set; ensembles moved AUC .875→.881 only.
  - **ens6m decode grid (smoother σ2–3 maps) + rules** (`night/scr_dec.py` 17×17 grid, honest finds F1): best finds F1 .732 (vs D1 .723). New C\* (`night/mkM_any.py`, q.9 A.25 D1.75):
    | C\* | geo P/R _ht | MagGr P/R _ht | honest Δ pce | accepted |
    |---|---|---|---|---|
    | E1_M3 | .644/.702 | .671/.732 | +0.0387 | yes |
    | E1_R3 | .668/.686 | .703/.712 | +0.0395 | yes |
    | E1_D1 (ens6 decode) | .664/.697 | .696/.725 | +0.0397 | yes |
    | E1_D2 (max R) | .654/.704 | .684/**.734** | +0.0395 | yes |
    | **E1_G1** (ens6m fs3 n0.5, X.6) | **.693**/.674 | **.736**/.697 | **+0.0407** | yes 5/5, p 1.0 |
    | E1_G2 (ens6m fs2 n1 p.005, X.7) | .676/.691 | .714/.719 | +0.0405 | yes 5/5 |
    | REF_v2 | .664/.785 | .671/.859 | — | — |
    sha8: G1 c92b25e0 (_ht aaba16ef), G2 4827920c (_ht 64256f6b). Gate JSON `w4geo/work/cstar/gate_CetO3_Sr8G_noTerFl_L_E1_{G1,G2,D1,D2,INS}{,_ht}.json`, honest `cstar/R_{G,D,INS}.json`. NB: G picked from 289 screen configs on the same folds → selection bias (plateau: top-10 all .728–.732).
  - Gate vs REF: **FAIL** for all (c_ge_ref_r: geo R and MagGr R). G1 passes P axes (geo P .693 ≥ .664, MagGr P .736 ≥ .671). Elena_ready: **no**. For A→C re-gate: **E1_G1** (main, best Δ), E1_G2 (balanced), E1_D2 (max MagGr R). Stop here: no remaining lever can move honest MagGr R_ht meaningfully; next real step would be a gate definition change (honest REF / OOF REF) or new data, not more grind. No LB.
- 30.10 23:04 MSK | **FREEZE E1_G1** (Elena «как лучше»). MagGr R vs REF wall accepted (OOF ceiling ~.73; REF in-sample). Package: `cstar/C_CetO3_Sr8G_noTerFl_L_E1_G1{,_ht}.geojson` sha c92b25e0/aaba16ef. Next PIXGBM only if new data or REF-OOF protocol.
