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
OR_FAMILIES = [("Кавказ", OR_CAUCASUS), ("Елизаветинское", ["033_Елизаветинское"]), ("Изборск", ["185_Изборск"])]
folds_or = {"family_of": {r: fam for fam, rs in OR_FAMILIES for r in rs},
            "folds": [{"fold": i, "test_families": [fam], "test_regions": rs} for i, (fam, rs) in enumerate(OR_FAMILIES)]}
_t = [r for f in folds_or["folds"] for r in f["test_regions"]]
assert sorted(_t) == sorted(OR_REGIONS) and len(_t) == len(set(_t)), "Or: каждый участок ровно в одном тестовом фолде"
assert all(len({folds_or["family_of"][r] for r in f["test_regions"]}) == 1 for f in folds_or["folds"]), "Or: фолд = одна семья"
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
VARIANTS += [("or_o2c", "full", "koz3-dfine-or", {"FOLDS_DEFAULT": "0,1,2", "DEADLINE_H_DEFAULT": 5.0, "FULL_ITERS_DEFAULT": 1200,
                                                "DFINE_BASE_DEFAULT": "obj2coco", "MOD_DEFAULT": "Or", "_folds": folds_or,
                                                "_kernel_sources": ["dmitrijkv/koz3-dfine-or-prep"]}, P_BASE + P_MOD),
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
