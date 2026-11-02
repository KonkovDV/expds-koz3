#!/usr/bin/env python3
"""P-W2-13: builds the D-FINE-2 kernels (second SpOr D-FINE ensemble member) from the bot-9 kernel
kaggle/2026-10-29/dfine_spor/full/koz3-dfine-spor-full.py by asserted text replacements.

What changes (all new behaviour sits behind flags whose code defaults reproduce the old kernel):
  * data: no S3 / no credentials. Rasters and GT come from private Kaggle datasets
    dmitrijkv/koz3-w2-dfine2-raw-{a,b} (flat files <region>__<file>.tif, gt_train_v2.geojson);
    DFINE_RAW (':'-separated dirs, default /kaggle/input) and DFINE_GT override for local runs.
  * DFINE_NUM_DN   (code default -1 = model default DN)  -> kernel 0 (DN off)
  * DFINE_SAMPLING (code default class)                  -> kernel site (region mass ~ n_r**DFINE_SITE_POW)
  * DFINE_QAUG2    (code default 0)                      -> kernel 1 (extra photometric quality augs)
  * DFINE_SEED_BASE (code default 100: model 100+k, batches 1000+k) -> kernel 200 (200+k, 2000+k)
  * DFINE_GROUP    (code default sp_dfine)               -> kernel sp_dfine2 (OOF property "group")
usage: python make_dfine2.py   (writes dmitrijkv/koz3-w2-dfine2-{a,b}/ next to this file)"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1] / "kaggle/2026-10-29/dfine_spor/full/koz3-dfine-spor-full.py"
KERNELS = {  # slug: (folds, deadline_h)
    "koz3-w2-dfine2-a": ("0,3", 4.0),
    "koz3-w2-dfine2-b": ("4,2,1", 5.0),
}
DATASETS = ["dmitrijkv/koz3-w2-dfine2-raw-a", "dmitrijkv/koz3-w2-dfine2-raw-b"]
DOCKER = "gcr.io/kaggle-private-byod/python@sha256:9a6bcfda6d03b2cd9aa347ee8c1bae62043328ed0f4afe9bda014e4f8bbf9435"


def rep(src, old, new, count=1):
    n = src.count(old)
    assert n == count, f"expected {count} occurrence(s), found {n}: {old[:80]!r}"
    return src.replace(old, new)


def cut(src, start, end, new):
    """Replace src[start_marker : end_marker) (end marker kept)."""
    i, j = src.index(start), src.index(end)
    assert src.count(start) == 1 and src.count(end) == 1 and i < j, (start[:60], end[:60])
    return src[:i] + new + src[j:]


HEADER = """# KOZ3 · W2 P-W2-13 D-FINE-2: second SpOr D-FINE-L ensemble member (built by w2_kernels/koz3-w2-dfine2/make_dfine2.py
# from bot-9 kaggle/2026-10-29/dfine_spor/full/koz3-dfine-spor-full.py). Same model/tiling/trainer/folds; differences are flags:
# DN off, per-site balanced sampling, extra photometric quality augs, seed base 200, OOF group sp_dfine2.
# Data: private Kaggle datasets dmitrijkv/koz3-w2-dfine2-raw-{a,b} (no credentials anywhere; no S3).
"""

STEP2 = '''# ── Шаг 2. Входы: растры SpOr и GT из приватных датасетов Kaggle (без ключей), веса HF (ранний фейл) ───────────────
def _inputs():
    roots = [Path(p) for p in os.environ.get("DFINE_RAW", "").split(":") if p] or [Path("/kaggle/input")]
    # Kaggle вырезает не-ASCII из имён файлов датасета, поэтому там лежат <NNN>__<kk>_SpOr.tif + manifest_*.json
    # {ascii_имя: {"region", "file"}}; локально (без манифеста) имя <участок>__<файл>.tif разбирается как есть.
    files, gts, man = {}, [], {}
    for r in roots:
        for p in sorted(r.rglob("*")):
            if not p.is_file(): continue
            if p.name.lower().endswith((".tif", ".tiff")) and "__" in p.name and "SpOr" in p.name: files.setdefault(p.name, p)
            elif p.name == "gt_train_v2.geojson": gts.append(p)
            elif p.name.startswith("manifest_") and p.suffix == ".json": man.update(json.load(open(p, encoding="utf-8")))
    lost = sorted(set(man) - set(files))
    if lost: raise SystemExit(f"файлы из manifest не найдены: {len(lost)} (напр. {lost[:3]}); датасеты {DATASETS_HINT}")
    tifs = {}  # путь → (участок, исходное имя файла)
    for n, p in files.items():
        tifs[str(p)] = (man[n]["region"], man[n]["file"]) if n in man else tuple(n.split("__", 1))
    gt = Path(os.environ["DFINE_GT"]) if os.environ.get("DFINE_GT") else (gts[0] if gts else None)
    return tifs, gt
TIF_INFO, GT_PATH = _inputs()
TIF_PATHS = sorted(Path(k) for k in TIF_INFO)
if not TIF_PATHS or GT_PATH is None or not GT_PATH.exists():
    raise SystemExit(f"входы не найдены: растров {len(TIF_PATHS)}, GT {GT_PATH} (датасеты {DATASETS_HINT})")
def _load_gt_all(p):
    from shapely.geometry import shape
    out = collections.defaultdict(list)
    for f in json.load(open(p, encoding="utf-8"))["features"]:
        q = f.get("properties") or {}; g = f.get("geometry")
        if q.get("markup_type") != "sp_or" or q.get("class_name") not in CLASSES or not g: continue
        if g.get("type") not in ("Polygon", "MultiPolygon"): continue
        out[q["region_name"]].append((shape(g), CLASSES.index(q["class_name"])))  # EPSG:3857
    return out
GT_BY = _load_gt_all(GT_PATH)
log(f"входы OK: растров {len(TIF_PATHS)}, GT {GT_PATH.name}: {sum(map(len, GT_BY.values()))} объектов sp_or, участков {len(GT_BY)}")
'''

STEP3_INV = '''def fetch(key, size):  # растр уже смонтирован датасетом Kaggle — путь как есть
    return Path(key)

def region_inv(region, paths):
    return {"region": region, "tifs": sorted((str(p), p.stat().st_size) for p in paths), "marks": bool(GT_BY.get(region)), "utm": None}

_by_reg = collections.defaultdict(list)
for _p in TIF_PATHS: _by_reg[TIF_INFO[str(_p)][0]].append(_p)
regions = sorted(_by_reg)
if CFG["regions"]: regions = [r for r in regions if r in CFG["regions"].split(",")]
INV = [region_inv(r, _by_reg[r]) for r in regions]
skipped = [i["region"] for i in INV if i["region"] not in FOLD]
INV = [i for i in INV if i["region"] in FOLD]
log(f"участков со SpOr в фолдах: {len(INV)} (с GT {sum(bool(i['marks']) for i in INV)}); вне фолдов (пропуск): {len(skipped)}")

'''

LOAD_GT = '''def load_gt(inv, crs):
    from shapely.ops import transform as st
    from pyproj import Transformer
    tr = Transformer.from_crs("EPSG:3857", crs, always_xy=True).transform; out = []
    for g, c in GT_BY.get(inv["region"], []):
        g = st(tr, g); g = g if g.is_valid else g.buffer(0)
        if not g.is_empty: out.append((g.bounds, c))
    return out

'''

WEIGHTS_OLD = '''        self.w = np.array([min(CFG["cls_cap"], math.sqrt(nmax / n[r["cls"]])) for r in self.obj], float)
        self.w = self.w / self.w.sum() if len(self.w) else self.w
'''
WEIGHTS_NEW = '''        self.w = np.array([min(CFG["cls_cap"], math.sqrt(nmax / n[r["cls"]])) for r in self.obj], float)
        self.wb = None
        if CFG["sampling"] == "site":  # P-W2-13: масса участка ∝ n_r**site_pow (классовые веса внутри участка сохранены); фон так же
            def _site(recs, w):
                nr = collections.Counter(r["region"] for r in recs); sr = collections.defaultdict(float)
                for r, x in zip(recs, w): sr[r["region"]] += x
                return np.array([x / sr[r["region"]] * nr[r["region"]] ** CFG["site_pow"] for r, x in zip(recs, w)], float)
            self.w = _site(self.obj, self.w) if len(self.w) else self.w
            self.wb = _site(self.bg, np.ones(len(self.bg))) if self.bg else None
            if self.wb is not None: self.wb = self.wb / self.wb.sum()
        self.w = self.w / self.w.sum() if len(self.w) else self.w
'''
BG_OLD = '''self.bg[rs.randint(len(self.bg))]'''
BG_NEW = '''self.bg[rs.randint(len(self.bg)) if self.wb is None else rs.choice(len(self.bg), p=self.wb)]'''

QAUG_OLD = '''                img = cv2.imdecode(enc, 1).astype(np.float32) + rs.normal(0, rs.uniform(0, 6), (T, T, 3))
'''
QAUG_NEW = QAUG_OLD + '''            if CFG["qaug2"]:  # P-W2-13: фотометрия качества снимка, без изменения геометрии (рамки не трогаются)
                img = np.clip(img, 0, 255)
                if rs.rand() < 0.2:  # дымка
                    a = rs.uniform(0.05, 0.3); img = img * (1 - a) + a * rs.uniform(150, 230) * rs.uniform(0.95, 1.05, 3)
                if rs.rand() < 0.2:  # снижение контраста
                    m = img.mean((0, 1), keepdims=True); img = m + (img - m) * rs.uniform(0.6, 0.9)
                if rs.rand() < 0.3:  # насыщенность / цветовой сдвиг
                    g = img.mean(2, keepdims=True); img = g + (img - g) * rs.uniform(0.5, 1.5) + rs.uniform(-12, 12, 3)
                if rs.rand() < 0.2:  # нерезкое маскирование (пересвеченная резкость)
                    img = img + rs.uniform(0.3, 1.0) * (img - cv2.GaussianBlur(img, (0, 0), rs.uniform(0.8, 2.0)))
                if rs.rand() < 0.25:  # JPEG на родном разрешении
                    ok, enc = cv2.imencode(".jpg", np.clip(img, 0, 255).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, int(rs.uniform(30, 85))])
                    img = cv2.imdecode(enc, 1).astype(np.float32)
'''


def build(src, folds, deadline):
    s = HEADER + src
    s = rep(s, '''# Ключи S3: Kaggle Secrets KOZ3_S3_* → env → /kaggle/input/**/cred.json (датасет dmitrijkv/koz3-s3-cred, поля id/key).
# Значения ключей никогда не печатаются и не пишутся в /kaggle/working.
''', "")
    s = rep(s, '"boto3": "boto3==1.40.61", ', "")
    s = rep(s, "DFINE_MODE_DEFAULT = 'full'\nFOLDS_DEFAULT = '4,2,1'\nDEADLINE_H_DEFAULT = 5.0\nFULL_ITERS_DEFAULT = 6000\n",
            f"DFINE_MODE_DEFAULT = 'full'\nFOLDS_DEFAULT = '{folds}'\nDEADLINE_H_DEFAULT = {deadline}\nFULL_ITERS_DEFAULT = 6000\n"
            "NUM_DN_DEFAULT = 0\nSAMPLING_DEFAULT = 'site'\nQAUG2_DEFAULT = 1\nSEED_BASE_DEFAULT = 200\nGROUP_DEFAULT = 'sp_dfine2'\n"
            f"DATASETS_HINT = {DATASETS!r}\n")
    s = rep(s, 'num_dn=E("DFINE_NUM_DN", -1)', 'num_dn=E("DFINE_NUM_DN", int(globals().get("NUM_DN_DEFAULT", -1)))')
    s = rep(s, '''           aug=os.environ.get("DFINE_AUG", "ours"))
''', '''           aug=os.environ.get("DFINE_AUG", "ours"),
           sampling=os.environ.get("DFINE_SAMPLING", globals().get("SAMPLING_DEFAULT", "class")), site_pow=E("DFINE_SITE_POW", 0.5),
           qaug2=E("DFINE_QAUG2", int(globals().get("QAUG2_DEFAULT", 0))), seed_base=E("DFINE_SEED_BASE", int(globals().get("SEED_BASE_DEFAULT", 100))),
           group=os.environ.get("DFINE_GROUP", globals().get("GROUP_DEFAULT", "sp_dfine")))
assert CFG["sampling"] in ("class", "site"), "DFINE_SAMPLING=class|site"
''')
    s = rep(s, '''CACHE = Path(os.environ.get("DFINE_CACHE", "/tmp/s3cache")); CHIPS = Path(os.environ.get("DFINE_CHIPS", "/tmp/chips"))
BUCKET, ENDPOINT = "train-expds-3", os.environ.get("DFINE_S3_ENDPOINT", "https://s3.ru-7.storage.selcloud.ru")
SECRET_ID, SECRET_KEY = "KOZ3_S3_ACCESS_KEY_ID", "KOZ3_S3_SECRET_ACCESS_KEY"
''', '''CHIPS = Path(os.environ.get("DFINE_CHIPS", "/tmp/chips"))
''')
    s = cut(s, "# ── Шаг 2. Секреты → S3", "try:\n    from huggingface_hub import snapshot_download", STEP2)
    s = cut(s, "def list_prefix(prefix, delim=None):", "def target_crs(inv, ds):", STEP3_INV)
    s = cut(s, "def load_gt(inv, crs):", "class Scene:", LOAD_GT)
    s = rep(s, WEIGHTS_OLD, WEIGHTS_NEW)
    s = rep(s, BG_OLD, BG_NEW)
    s = rep(s, QAUG_OLD, QAUG_NEW)
    s = rep(s, '"group": "sp_dfine",', '"group": CFG["group"],')
    s = rep(s, '"tif": key.rsplit("/", 1)[-1]}', '"tif": TIF_INFO[key][1] if key in TIF_INFO else key.rsplit("/", 1)[-1]}')
    s = rep(s, "make_model(seed=100 + k)", 'make_model(seed=CFG["seed_base"] + k)')
    s = rep(s, "batches(tr, 1000 + k, dev)", 'batches(tr, CFG["seed_base"] * 10 + k, dev)')
    for bad in ("boto3", "s3_client", "cred.json", "SECRET_KEY", "list_prefix", "CLI."):
        assert bad not in s, f"leftover {bad}"
    compile(s, "dfine2", "exec")
    return s


def main():
    src = BASE.read_text(encoding="utf-8")
    for slug, (folds, dl) in KERNELS.items():
        d = HERE / "dmitrijkv" / slug; d.mkdir(parents=True, exist_ok=True)
        (d / f"{slug}.py").write_text(build(src, folds, dl), encoding="utf-8")
        meta = {"id": f"dmitrijkv/{slug}", "title": slug, "code_file": f"{slug}.py", "language": "python", "kernel_type": "script",
                "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": True, "keywords": ["gpu"],
                "dataset_sources": DATASETS, "kernel_sources": [], "competition_sources": [], "model_sources": [],
                "docker_image": DOCKER, "machine_shape": "NvidiaTeslaT4"}
        (d / "kernel-metadata.json").write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8")
        print("wrote", d)


if __name__ == "__main__":
    main()
