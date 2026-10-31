# Сборка двух ядер из src/dfine_main.py: smoke (DFINE_MODE=smoke, 10 ит.) и full. Фолды — те же, что у верификатора.
import json, pathlib
H = pathlib.Path(__file__).parent
F = json.load(open(H.parent / "nolidar_verifier/folds_family_k5.json", encoding="utf-8"))
folds = {"family_of": F["family_of"], "folds": [{k: f[k] for k in ("fold", "test_families", "test_regions")} for f in F["folds"]]}
src = (H / "src/dfine_main.py").read_text(encoding="utf-8")
# Тренер бота 5 (адаптер, вендорено из bot-5/repo@0256251) встраивается строкой: Kaggle загружает только файл кода.
bot5 = (H / "src/dfine_train_bot5.py").read_text(encoding="utf-8")
assert "'''" not in bot5 and not bot5.endswith("\\"), "dfine_train_bot5.py нельзя встроить в r'''…'''"
import hashlib; print("dfine_train_bot5.py sha256", hashlib.sha256(bot5.encode("utf-8")).hexdigest())
head, rest = src.split("import json, time, re", 1)
# Бюджет недели (решение 22:26): full = 3 фолда 4,2,1 (2 GPU: 4 ‖ 2, затем 1), потолок 5 ч; full_rest = 0,3 на следующую неделю.
# 6000 ит. на фолд: 0,92 с/ит (смоук v2, T4) → 1,53 ч обучения; см. DFINE.md §11.
# Патчи применяются ТОЛЬКО к новым вариантам — smoke/full/full_rest собираются из src байт-в-байт как раньше.
HF_LINE = 'HF_REPO, HF_REV = "ustc-community/dfine-large-coco", "aa8f8211545e4cff5adf52ea512f9ac74a9824ec"\n'
P_BASE = [(HF_LINE, 'HF_BASES = {"coco": ("ustc-community/dfine-large-coco", "aa8f8211545e4cff5adf52ea512f9ac74a9824ec"),  # как фолды full/full_rest\n'
                    '            "obj2coco": ("ustc-community/dfine-large-obj2coco-e25", "c89de47153e338c07cddc81fc21a776b97cbf75e")}  # Objects365→COCO, Apache-2.0\n'),
          ('MODE = os.environ.get("DFINE_MODE", DFINE_MODE_DEFAULT)\n',
           'MODE = os.environ.get("DFINE_MODE", DFINE_MODE_DEFAULT)\n'
           'DFINE_BASE = os.environ.get("DFINE_BASE", globals().get("DFINE_BASE_DEFAULT", "coco")); assert DFINE_BASE in HF_BASES, "DFINE_BASE=coco|obj2coco"\n'
           'HF_REPO, HF_REV = HF_BASES[DFINE_BASE]; SP_GROUP = "sp_dfine" if DFINE_BASE == "coco" else "sp_dfine_o2c"\n'),
          ('"original_crs": sc.crs, "group": "sp_dfine",', '"original_crs": sc.crs, "group": SP_GROUP, "threshold_group": SP_GROUP,'),
          ('REPORT, lock = {"cfg": CFG, "hf": f"{HF_REPO}@{HF_REV}",', 'REPORT, lock = {"cfg": CFG, "hf": f"{HF_REPO}@{HF_REV}", "base": DFINE_BASE,')]
FINAL_SRC = (H / "src/final_step8.py").read_text(encoding="utf-8")
STEP8 = "# ── Шаг 8. Оркестрация"
P_FINAL = [("# лишь после письма организаторов о весах. Режимы: DFINE_MODE = smoke (10 ит., мини-данные) | full.\n",
            "# лишь после письма организаторов о весах. Режимы: DFINE_MODE = smoke (10 ит., мини-данные) | full | final.\n"
            "# final (29.10): одна модель на ВСЕХ участках SpOr → /kaggle/working/dfine_final/ (save_pretrained, fp32) + замер скорости; без OOF.\n"),
           (None, FINAL_SRC)]  # None: заменить всё от «Шаг 8» до конца
def apply(text, patches):
    for old, new in patches:
        if old is None:
            i = text.index(STEP8); assert text.count(STEP8) == 1; text = text[:i] + new
        else:
            assert text.count(old) == 1, ("патч не найден ровно 1 раз", old[:60]); text = text.replace(old, new)
    return text
VARIANTS = [("smoke", "smoke", "koz3-dfine-spor-smoke", {}, []),
            ("full", "full", "koz3-dfine-spor-full", {"FOLDS_DEFAULT": "4,2,1", "DEADLINE_H_DEFAULT": 5.0, "FULL_ITERS_DEFAULT": 6000}, []),
            ("full_rest", "full", "koz3-dfine-spor-full-rest", {"FOLDS_DEFAULT": "0,3", "DEADLINE_H_DEFAULT": 3.5, "FULL_ITERS_DEFAULT": 6000}, []),
            # 29.10: финальная модель (COCO, как фолды) и опциональные фолды obj2coco (своя OOF, threshold_group sp_dfine_o2c)
            ("final", "final", "koz3-dfine-spor-final", {"DEADLINE_H_DEFAULT": 3.0, "FULL_ITERS_DEFAULT": 6000, "DFINE_BASE_DEFAULT": "coco"}, P_BASE + P_FINAL),
            ("full_o2c", "full", "koz3-dfine-spor-o2c-a", {"FOLDS_DEFAULT": "4,2,1", "DEADLINE_H_DEFAULT": 5.0, "FULL_ITERS_DEFAULT": 6000, "DFINE_BASE_DEFAULT": "obj2coco"}, P_BASE),
            ("full_rest_o2c", "full", "koz3-dfine-spor-o2c-b", {"FOLDS_DEFAULT": "0,3", "DEADLINE_H_DEFAULT": 3.5, "FULL_ITERS_DEFAULT": 6000, "DFINE_BASE_DEFAULT": "obj2coco"}, P_BASE)]
# 29.10 (задача B): D-FINE на Or (ортофото) и Ae (аэрофото), база obj2coco. Патчи P_MOD поверх P_BASE.
PREP_LIB = (H / "src/or_prep_lib.py").read_text(encoding="utf-8")
OR_REGIONS = ["008_Герите", "014_Лешки", "033_Елизаветинское", "056_Фалхан", "093_Старый_Барах", "185_Изборск"]
# Or (решение bot 1, 29.10 07:43): 4 кавказских участка «архитектуры» лежат в ≤25 км друг от друга (центроиды GT,
# or_gt_counts.log) → leave-one-site-out оптимистичен. Они объединены в ОДНУ семью «Кавказ»; Or = leave-one-family-out, 3 фолда:
#   f0 = Кавказ (008_Герите, 014_Лешки, 056_Фалхан, 093_Старый_Барах), f1 = 033_Елизаветинское, f2 = 185_Изборск.
OR_CAUCASUS = ["008_Герите", "014_Лешки", "056_Фалхан", "093_Старый_Барах"]
# P_OR_FIX (бот 9, 29.10 14:00, задача «Or-fix»): 3 фолда по 6 участкам так, чтобы в train каждого фолда были архитектуры,
# городища и (где возможно) фортификации. Фортификации есть ТОЛЬКО в 185_Изборск → в фолде теста Изборска их в train нет (неизбежно).
# Ядро Кавказа (Герите–Старый Барах 1,9 км, Лешки 6,1–8,0 км) — одна семья и один тестовый фолд; Фалхан (17,7–24,9 км от ядра)
# — отдельная семья (компромисс: иначе фолд Кавказа не имеет архитектур в train). Подробно: DFINE_FINAL_ORAE.md §11.
#   f0 = Кавказ-ядро (008_Герите, 014_Лешки, 093_Старый_Барах); f1 = Фалхан + Елизаветинское; f2 = Изборск.
OR_FOLDS = [[("Кавказ-ядро", ["008_Герите", "014_Лешки", "093_Старый_Барах"])],
            [("Фалхан", ["056_Фалхан"]), ("Елизаветинское", ["033_Елизаветинское"])],
            [("Изборск", ["185_Изборск"])]]
folds_or = {"family_of": {r: fam for fl in OR_FOLDS for fam, rs in fl for r in rs},
            "folds": [{"fold": i, "test_families": [fam for fam, _ in fl], "test_regions": [r for _, rs in fl for r in rs]}
                      for i, fl in enumerate(OR_FOLDS)]}
_t = [r for f in folds_or["folds"] for r in f["test_regions"]]
assert sorted(_t) == sorted(OR_REGIONS) and len(_t) == len(set(_t)), "Or: каждый участок ровно в одном тестовом фолде"
_ff = [fam for f in folds_or["folds"] for fam in f["test_families"]]
assert len(_ff) == len(set(_ff)), "Or: семья не может быть в двух фолдах"
P_MOD = [('HF_REPO, HF_REV = HF_BASES[DFINE_BASE]; SP_GROUP = "sp_dfine" if DFINE_BASE == "coco" else "sp_dfine_o2c"\n',
          'HF_REPO, HF_REV = HF_BASES[DFINE_BASE]; SP_GROUP = "sp_dfine" if DFINE_BASE == "coco" else "sp_dfine_o2c"\n'
          'MOD = os.environ.get("DFINE_MOD", globals().get("MOD_DEFAULT", "SpOr")); assert MOD in ("SpOr", "Or", "Ae"), "DFINE_MOD=SpOr|Or|Ae"\n'
          'MARKUP_TYPE = {"SpOr": "sp_or", "Or": "or", "Ae": "ae"}[MOD]\n'
          'if MOD != "SpOr": SP_GROUP = MOD.lower() + "_dfine" + ("" if DFINE_BASE == "obj2coco" else "_coco")  # or_dfine / ae_dfine\n'),
         ('elif "_SpOr_" in k and n.lower().endswith((".tif", ".tiff")): tifs.append((k, s))',
          'elif f"_{MOD}_" in k and n.lower().endswith((".tif", ".tiff")): tifs.append((k, s))'),
         ('m = re.search(r"_SpOr_(.+)\\.geojson$", n)', 'm = re.search(rf"_{MOD}_(.+)\\.geojson$", n)'),
         ('class Scene:\n', PREP_LIB +
          'PREP_DIR = os.environ.get("DFINE_PREP_DIR", "/kaggle/input")\n'
          'def raster_path(key, size):\n'
          '    if MOD != "Or" or size < PREP_MIN_BYTES: return fetch(key, size)\n'
          '    import shutil; nm = prep_name(key, size)\n'
          '    for p in sorted(glob.glob(f"{PREP_DIR}/**/or_prep/{nm}", recursive=True)): return Path(p)  # из ядра koz3-dfine-or-prep\n'
          '    dst = CACHE / "or_prep" / nm\n'
          '    with PREP_LOCK:\n'
          '        if dst.exists(): return dst\n'
          '        free = shutil.disk_usage(CACHE.parent if CACHE.parent.exists() else "/").free\n'
          '        log(f"[prep] нет {nm} из koz3-dfine-or-prep — скачиваю {size / 1e9:.1f} ГБ в ядре (свободно {free / 1e9:.0f} ГБ)")\n'
          '        if free < size * 1.02 + 2e9: raise RuntimeError("мало диска для " + nm)\n'
          '        t = time.time(); src = fetch(key, size); log(f"[prep] скачано за {time.time() - t:.0f} с ({size / 1e6 / max(1, time.time() - t):.0f} МБ/с)")\n'
          '        try: prep_tif(src, dst, log)\n'
          '        finally: src.unlink(missing_ok=True)\n'
          '    return dst\n\n'
          'class Scene:\n'),
         ('self.src = rasterio.open(fetch(key, size)); self.crs', 'self.src = rasterio.open(raster_path(key, size)); self.crs'),
         ('sc = Scene(inv, key, size); gt = load_gt(inv, sc.crs)\n', 'sc = Scene(inv, key, size); gt = load_gt(inv, sc.crs); _org = []\n'),
         ('                rec.append({"path": str(p), "region"', '                _org.append((x0, y0)); rec.append({"path": str(p), "region"'),
         ('            sc.close()\n        except Exception as e:\n            log("чипы: ошибка"',
          '            # проверка покрытия GT: рамка целиком внутри хотя бы одного сохранённого чипа (BIG×BIG до сдвига ±1/3)\n'
          '            _cov = sum(any(b[0] >= x and b[1] >= y and b[2] <= x + BIG and b[3] <= y + BIG for x, y in _org) for b, c in boxes)\n'
          '            COVER.append({"region": inv["region"], "tif": key.rsplit("/", 1)[-1], "W": sc.W, "H": sc.H, "gt_total": len(gt),\n'
          '                          "gt_in_raster": len(boxes), "gt_covered_by_chip": _cov, "chips": len(_org)})\n'
          '            log("покрытие GT:", json.dumps(COVER[-1], ensure_ascii=False))\n'
          '            sc.close()\n        except Exception as e:\n            log("чипы: ошибка"'),
         ('def build_chips(inv):\n', 'COVER = []\ndef build_chips(inv):\n'),
         ('"geometry": mapping(st(to3857, Polygon(ring))),', '"geometry": mapping(_valid(st(to3857, Polygon(ring)))),'),
         ('@torch.no_grad()\ndef infer_region(', 'def _valid(g): return g if g.is_valid else g.buffer(0)\n@torch.no_grad()\ndef infer_region('),
         ('"markup_type": "sp_or",', '"markup_type": MARKUP_TYPE,'),
         ('"base": DFINE_BASE,', '"base": DFINE_BASE, "mod": MOD, "markup_type": MARKUP_TYPE, "gt_coverage": COVER,'),
         ('open(OUT / "oof_dfine_spor.geojson", "w"', 'open(OUT / f"oof_dfine_{MOD.lower()}.geojson", "w"'),
         ('# лишь после письма организаторов о весах. Режимы: DFINE_MODE = smoke (10 ит., мини-данные) | full.\n',
          '# лишь после письма организаторов о весах. Режимы: DFINE_MODE = smoke (10 ит., мини-данные) | full.\n'
          '# 29.10 вариант Or/Ae (DFINE_MOD): та же схема на модальности Or/Ae, база obj2coco, threshold_group or_dfine/ae_dfine.\n')]
# 29.10 11:50 (бот 1): Or — тайл 1024, перекрытие 288 (28 %), bs 4 (VRAM: 6,77 ГБ при 640/bs8 → ≈8,7 ГБ [оц, ×2,56/2]), ITERS 2400 (= 9600 чипов, как 1200×8).
P_OR1024 = [('tile=E("DFINE_TILE", 640), overlap=E("DFINE_OVERLAP", 160)', 'tile=E("DFINE_TILE", 1024), overlap=E("DFINE_OVERLAP", 288)'),
            ('bs=E("DFINE_BS", 8)', 'bs=E("DFINE_BS", 4)')]
# P_OR_FIX (бот 9, 29.10): (1) корень Герите 1×1 px — UTM.json 008_Герите задаёт EPSG:4326 (градусы), Scene варпил с
# resolution=0,5 ГРАДУСА → 1×1. Теперь CRS из UTM.json берётся, только если она проекционная, иначе — UTM по центру растра.
# (2) явные негативные чипы T×T без GT (cls −2, boxes []) ≈1:1 к объектам участка; (3) сэмплер obj/neg/bg = 0,5/0,3/0,2,
# cls_cap 6 (редкие классы); (4) лог состава train каждого фолда; DFINE_CHIPS_ONLY=1 — только банк чипов (CPU-дриран).
P_OR_FIX = [
    ('        if m: return f"EPSG:{m.group(1)}"\n',
     '        if m:\n'
     '            from pyproj import CRS as _CRS\n'
     '            if _CRS.from_user_input(f"EPSG:{m.group(1)}").is_projected: return f"EPSG:{m.group(1)}"\n'
     '            log(f"[crs] {inv[\'region\']}: UTM.json задаёт EPSG:{m.group(1)} (географическая, градусы) — беру UTM по центру растра")\n'),
    ('obj_frac=0.6, cls_cap=4.0,',
     'obj_frac=E("DFINE_OBJ_FRAC", 0.5), cls_cap=E("DFINE_CLS_CAP", 6.0), neg_share=E("DFINE_NEG_SHARE", 0.6),\n'
     '           neg_ratio=E("DFINE_NEG_RATIO", 1.0), neg_min=E("DFINE_NEG_MIN", 8), neg_cap=E("DFINE_NEG_CAP", 60), neg_margin=E("DFINE_NEG_MARGIN", 32),\n'
     '           neg_valid=E("DFINE_NEG_VALID", 0.25), neg_masked=E("DFINE_NEG_MASKED", 1), neg_mask_max=E("DFINE_NEG_MASK_MAX", 0.4),'),
    ('COVER = []\ndef build_chips(inv):\n', 'COVER, NEG = [], []\ndef build_chips(inv):\n'),
    ('            # проверка покрытия GT: рамка целиком внутри',
     '            # P_OR_FIX: явные негативы — окна T×T, не задевающие ни одну рамку GT (+поле neg_margin px); окно может выходить за\n'
     '            # край растра (как окна инференса), но валидных пикселей ≥ neg_valid·T²; окна сдвинуты друг от друга ≥ T/3 по x или y.\n'
     '            # Сколько: clip(round(neg_ratio·объектов участка), neg_min, neg_cap). Обучение на них: пустые цели (проверено: HF D-FINE\n'
     '            # даёт конечный loss только из VFL, бот-9/or_fix/test_empty_targets.log).\n'
     '            T, M = CFG["tile"], CFG["neg_margin"]\n'
     '            nneg = 0 if CFG["neg_ratio"] <= 0 else min(CFG["neg_cap"], max(CFG["neg_min"], int(round(CFG["neg_ratio"] * len(objs)))))\n'
     '            got, tries, negw = 0, 0, []\n'
     '            while got < nneg and tries < 60 * max(1, nneg):\n'
     '                tries += 1; x0, y0 = int(rng.uniform(-T / 2, sc.W - T / 2)), int(rng.uniform(-T / 2, sc.H - T / 2))\n'
     '                if any(b[2] + M > x0 and b[3] + M > y0 and b[0] - M < x0 + T and b[1] - M < y0 + T for b, _c in boxes): continue\n'
     '                if any(abs(x0 - u) < T / 3 and abs(y0 - v) < T / 3 for u, v in negw): continue\n'
     '                img = sc.read(x0, y0, T)\n'
     '                if (img.max(2) > 0).mean() < CFG["neg_valid"]: drop["neg_nodata"] += 1; continue\n'
     '                negw.append((x0, y0)); got += 1\n'
     '                p = CHIPS / f"{inv[\'region\']}__{Path(key).stem}__neg{got}.jpg"; p.parent.mkdir(parents=True, exist_ok=True)\n'
     '                cv2.imwrite(str(p), img[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 95])\n'
     '                rec.append({"path": str(p), "region": inv["region"], "fold": FOLD[inv["region"]], "cls": -2, "boxes": [], "c": None})\n'
     '            # запас: если чистых окон не хватило (малые растры Изборск/Елизаветинское: окно 1024 не помещается вне GT) — «маскированные»\n'
     '            # негативы: окно с GT, рамки GT (+поле) залиты 0 = nodata (как пустые края растра на инференсе); замазано ≤ neg_mask_max окна.\n'
     '            gotm, tries2 = 0, 0\n'
     '            while CFG["neg_masked"] and got + gotm < nneg and tries2 < 60 * max(1, nneg):\n'
     '                tries2 += 1; x0, y0 = int(rng.uniform(-T / 2, sc.W - T / 2)), int(rng.uniform(-T / 2, sc.H - T / 2))\n'
     '                if any(abs(x0 - u) < T / 3 and abs(y0 - v) < T / 3 for u, v in negw): continue\n'
     '                img = sc.read(x0, y0, T); mk = np.zeros((T, T), bool)\n'
     '                for b, _c in boxes:\n'
     '                    a0, b0 = max(0, int(b[0] - M - x0)), max(0, int(b[1] - M - y0)); a1, b1 = min(T, int(math.ceil(b[2] + M - x0))), min(T, int(math.ceil(b[3] + M - y0)))\n'
     '                    if a1 > a0 and b1 > b0: mk[b0:b1, a0:a1] = True\n'
     '                if not mk.any() or mk.mean() > CFG["neg_mask_max"]: continue\n'
     '                img[mk] = 0\n'
     '                if (img.max(2) > 0).mean() < CFG["neg_valid"]: drop["negm_nodata"] += 1; continue\n'
     '                negw.append((x0, y0)); gotm += 1\n'
     '                p = CHIPS / f"{inv[\'region\']}__{Path(key).stem}__negm{gotm}.jpg"; p.parent.mkdir(parents=True, exist_ok=True)\n'
     '                cv2.imwrite(str(p), img[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 95])\n'
     '                rec.append({"path": str(p), "region": inv["region"], "fold": FOLD[inv["region"]], "cls": -2, "boxes": [], "c": None, "masked": True})\n'
     '            NEG.append({"region": inv["region"], "tif": key.rsplit("/", 1)[-1], "W": sc.W, "H": sc.H, "want": nneg, "got_clean": got,\n'
     '                        "got_masked": gotm, "tries": tries + tries2})\n'
     '            log("негативы:", json.dumps(NEG[-1], ensure_ascii=False))\n'
     '            # проверка покрытия GT: рамка целиком внутри'),
    ('self.obj = [r for r in recs if r["cls"] >= 0]; self.bg = [r for r in recs if r["cls"] < 0]',
     'self.obj = [r for r in recs if r["cls"] >= 0]; self.bg = [r for r in recs if r["cls"] == -1]; self.neg = [r for r in recs if r["cls"] == -2]'),
    ('            use_obj = self.obj and (not self.bg or rs.rand() < CFG["obj_frac"])\n'
     '            r = self.obj[rs.choice(len(self.obj), p=self.w)] if use_obj else self.bg[rs.randint(len(self.bg))]\n',
     '            use_obj = self.obj and (not (self.bg or self.neg) or rs.rand() < CFG["obj_frac"])\n'
     '            if use_obj: r = self.obj[rs.choice(len(self.obj), p=self.w)]\n'
     '            else:  # P_OR_FIX: не-объектный чип — негатив с вероятностью neg_share, иначе случайный фон\n'
     '                pool = self.neg if self.neg and (not self.bg or rs.rand() < CFG["neg_share"]) else self.bg\n'
     '                r = pool[rs.randint(len(pool))]\n'),
    ('if not RECS: raise SystemExit("пустой банк чипов — проверьте разметку/растры")\n',
     'if not RECS: raise SystemExit("пустой банк чипов — проверьте разметку/растры")\n'
     'FOLD_CHIPS = {}  # P_OR_FIX: состав train каждого фолда (классы, фон, негативы)\n'
     'for k in folds_run:\n'
     '    _tr = [r for r in RECS if r["fold"] != k]\n'
     '    FOLD_CHIPS[k] = {"test": sorted(i["region"] for i in INV if FOLD[i["region"]] == k), "train_chips": len(_tr),\n'
     '                     "obj": dict(collections.Counter(CLASSES[r["cls"]] for r in _tr if r["cls"] >= 0)),\n'
     '                     "bg": sum(r["cls"] == -1 for r in _tr), "neg": sum(r["cls"] == -2 for r in _tr),\n'
     '                     "neg_masked": sum(bool(r.get("masked")) for r in _tr),\n'
     '                     "neg_by_region": dict(collections.Counter(r["region"] for r in _tr if r["cls"] == -2))}\n'
     '    log(f"[фолд {k}] train:", json.dumps(FOLD_CHIPS[k], ensure_ascii=False))\n'
     '    _miss = [c for c in ("gorodishcha", "fortifikatsii", "arkhitektury") if not FOLD_CHIPS[k]["obj"].get(c)]\n'
     '    if _miss: log(f"[фолд {k}] ВНИМАНИЕ: в train нет классов {_miss}")\n'
     'if os.environ.get("DFINE_CHIPS_ONLY") == "1":\n'
     '    json.dump({"cfg": CFG, "gt_coverage": COVER, "negatives": NEG, "fold_chips": FOLD_CHIPS, "n_chips": len(RECS)},\n'
     '              open(OUT / "chips_report.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)\n'
     '    log("DFINE_CHIPS_ONLY=1: собран только банк чипов, обучение пропущено"); raise SystemExit(0)\n'),
    ('"markup_type": MARKUP_TYPE, "gt_coverage": COVER,', '"markup_type": MARKUP_TYPE, "gt_coverage": COVER, "negatives": NEG, "fold_chips": FOLD_CHIPS,'),
    ('# 29.10 вариант Or/Ae (DFINE_MOD): та же схема на модальности Or/Ae, база obj2coco, threshold_group or_dfine/ae_dfine.\n',
     '# 29.10 вариант Or/Ae (DFINE_MOD): та же схема на модальности Or/Ae, база obj2coco, threshold_group or_dfine/ae_dfine.\n'
     '# 29.10 P_OR_FIX (Or): CRS из UTM.json только проекционная (Герите 1×1 px), фолды Кавказ-ядро | Фалхан+Елизаветинское | Изборск,\n'
     '# негативные чипы, сэмплер obj/neg/bg 0,5/0,3/0,2, cls_cap 6. См. bot-9/DFINE_FINAL_ORAE.md §11.\n')]
VARIANTS += [("or_o2c", "full", "koz3-dfine-or", {"FOLDS_DEFAULT": "0,1,2", "DEADLINE_H_DEFAULT": 5.0, "FULL_ITERS_DEFAULT": 2400,
                                                "DFINE_BASE_DEFAULT": "obj2coco", "MOD_DEFAULT": "Or", "_folds": folds_or,
                                                "_kernel_sources": ["dmitrijkv/koz3-dfine-or-prep"]}, P_BASE + P_MOD + P_OR1024 + P_OR_FIX),
             ("ae_o2c", "full", "koz3-dfine-ae", {"FOLDS_DEFAULT": "3,4,0,2,1", "DEADLINE_H_DEFAULT": 5.5, "FULL_ITERS_DEFAULT": 4000,
                                                "DFINE_BASE_DEFAULT": "obj2coco", "MOD_DEFAULT": "Ae"}, P_BASE + P_MOD)]
for d, mode, name, ov, pt in VARIANTS:
    emb = f"DFINE_MODE_DEFAULT = {mode!r}\n" + "".join(f"{k} = {v!r}\n" for k, v in ov.items() if not k.startswith("_"))
    emb += "FOLDS = __import__('json').loads(r'''" + json.dumps(ov.get("_folds", folds), ensure_ascii=False) + "''')\n"
    emb += "BOT5_TRAINER_SRC = r'''" + bot5 + "'''\n"
    (H / d).mkdir(exist_ok=True)
    (H / d / f"{name}.py").write_text(apply(head + emb + "import json, time, re" + rest, pt), encoding="utf-8")
    md = {"id": f"dmitrijkv/{name}", "title": name, "code_file": f"{name}.py", "language": "python", "kernel_type": "script",
          "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": True, "machine_shape": "NvidiaTeslaT4",
          "dataset_sources": ["dmitrijkv/koz3-s3-cred"], "competition_sources": [], "kernel_sources": ov.get("_kernel_sources", []), "model_sources": [],
          "docker_image": "gcr.io/kaggle-private-byod/python@sha256:9a6bcfda6d03b2cd9aa347ee8c1bae62043328ed0f4afe9bda014e4f8bbf9435"}
    json.dump(md, open(H / d / "kernel-metadata.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(d, name, len((H / d / f"{name}.py").read_bytes()), "байт", hashlib.sha256((H / d / f"{name}.py").read_bytes()).hexdigest()[:16])
# CPU-ядро koz3-dfine-or-prep (без GPU): S3-клиент — дословно из src/dfine_main.py, подготовка — src/or_prep_lib.py
_m = src[src.index("def _cred_from_inputs():"):src.index("CLI = s3_client()")]
prep_src = (H / "src/or_prep_main.py").read_text(encoding="utf-8").replace("#@@S3@@\n", _m).replace("#@@PREP@@\n", PREP_LIB)
(H / "or_prep").mkdir(exist_ok=True); (H / "or_prep/koz3-dfine-or-prep.py").write_text(prep_src, encoding="utf-8")
json.dump({"id": "dmitrijkv/koz3-dfine-or-prep", "title": "koz3-dfine-or-prep", "code_file": "koz3-dfine-or-prep.py", "language": "python",
           "kernel_type": "script", "is_private": True, "enable_gpu": False, "enable_tpu": False, "enable_internet": True,
           "dataset_sources": ["dmitrijkv/koz3-s3-cred"], "competition_sources": [], "kernel_sources": [], "model_sources": [],
           "docker_image": "gcr.io/kaggle-private-byod/python@sha256:9a6bcfda6d03b2cd9aa347ee8c1bae62043328ed0f4afe9bda014e4f8bbf9435"},
          open(H / "or_prep/kernel-metadata.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("or_prep koz3-dfine-or-prep", len(prep_src.encode()), "байт", hashlib.sha256(prep_src.encode()).hexdigest()[:16])
