# ==== mn_lib: общие функции репро MataNerdy (S3, разметка, кропы по рецепту конкурента) ====
# Рецепт: competitors/MataNerdy-Geodata_Archaeology_CV/01_geodata_to_cv/build_multiclass_dataset.py
# (адаптивный кроп вокруг объекта, канал 1 растра, маска по площади убыв.), 03_.../arch_datasets/archaeology_dataset.py
# (bilinear/nearest resize до 256, z-score на патч). Ключи S3 — только Kaggle Secrets, не печатаются и не пишутся.
import os, re, json, time, math, threading, logging
from pathlib import Path
import numpy as np

BUCKET = "train-expds-3"
ENDPOINT = "https://s3.ru-7.storage.selcloud.ru"
SECRET_ID, SECRET_KEY = "KOZ3_S3_ACCESS_KEY_ID", "KOZ3_S3_SECRET_ACCESS_KEY"
CACHE = Path(os.environ.get("MN_CACHE", "/tmp/s3cache"))
CLASSES = ["background", "kurgany_tselye", "kurgany_povrezhdennye", "gorodishcha", "fortifikatsii", "arkhitektury"]
RU2CLS = {"курганы_целые": "kurgany_tselye", "курганы_поврежденные": "kurgany_povrezhdennye",
          "городища": "gorodishcha", "фортификации": "fortifikatsii", "архитектуры": "arkhitektury"}
MOD2MARKUP = {"Li": "li", "Ae": "ae", "SpOr": "sp_or"}
log = logging.getLogger("mn")


def s3_client():
    """boto3-клиент: ключи из Kaggle Secrets (на box/тестах — из окружения). Значения никуда не выводятся."""
    import boto3
    from botocore.config import Config
    kid = sec = None
    try:
        from kaggle_secrets import UserSecretsClient
        u = UserSecretsClient(); kid, sec = u.get_secret(SECRET_ID), u.get_secret(SECRET_KEY)
    except Exception:
        kid, sec = os.environ.get("AWS_ACCESS_KEY_ID"), os.environ.get("AWS_SECRET_ACCESS_KEY")
    if not kid or not sec:
        raise SystemExit(f"нет ключей S3: заведите Kaggle Secrets {SECRET_ID} и {SECRET_KEY} и привяжите к ядру")
    ep = os.environ.get("MN_S3_ENDPOINT", ENDPOINT)
    return boto3.client("s3", endpoint_url=ep, aws_access_key_id=kid, aws_secret_access_key=sec,
                        region_name=os.environ.get("AWS_DEFAULT_REGION", "ru-7"),
                        config=Config(retries={"max_attempts": 10, "mode": "adaptive"}, max_pool_connections=16))


def list_region(cli, region):
    out, tok = [], None
    while True:
        kw = {"Bucket": BUCKET, "Prefix": region + "/"}
        if tok: kw["ContinuationToken"] = tok
        r = cli.list_objects_v2(**kw)
        out += [(o["Key"], o["Size"]) for o in r.get("Contents", [])]
        if not r.get("IsTruncated"): return out
        tok = r["NextContinuationToken"]


def fetch(cli, key, size=None, tries=5):
    """Скачать ключ в кэш /tmp (докачка не нужна: проверка размера; повторы с паузой)."""
    from boto3.s3.transfer import TransferConfig
    dst = CACHE / key
    if dst.exists() and (size is None or dst.stat().st_size == size):
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    cfg = TransferConfig(multipart_threshold=32 << 20, multipart_chunksize=16 << 20, max_concurrency=8)
    for a in range(tries):
        try:
            cli.download_file(BUCKET, key, str(tmp), Config=cfg)
            tmp.rename(dst); return dst
        except Exception as e:
            log.warning("S3 повтор %d %s: %s", a + 1, key, type(e).__name__); time.sleep(5 * (a + 1))
    raise RuntimeError(f"не скачан {key}")


def region_files(keys):
    """Из листинга участка: растры по модальностям (Li — только *_Li_g.tif, как у конкурента), разметка, UTM.json."""
    rast, marks, utm = {"Li": [], "Ae": [], "SpOr": []}, [], None
    for k, s in keys:
        n = k.rsplit("/", 1)[-1]
        if n == "UTM.json": utm = (k, s)
        elif k.lower().endswith((".tif", ".tiff")):
            if "_Li_" in k and n.lower().endswith("_g.tif"): rast["Li"].append((k, s))
            elif "_Ae_" in k or "_Ае_" in k: rast["Ae"].append((k, s))
            elif "_SpOr_" in k: rast["SpOr"].append((k, s))
        elif n.endswith(".geojson") and "разметка" in k:
            m = re.search(r"_(Li|Ae|Ае|SpOr)_(.+)\.geojson$", n)
            if m and m.group(2) in RU2CLS:
                marks.append((k, s, {"Ае": "Ae"}.get(m.group(1), m.group(1)), RU2CLS[m.group(2)]))
    for m in rast: rast[m].sort()
    if rast["Ae"]: rast["Ae"] = rast["Ae"][:1]  # конкурент: choose_raster_for_modality → первый tif
    return rast, marks, utm


def utm_crs(path):
    d = json.load(open(path, encoding="utf-8"))
    m = re.search(r"EPSG::?(\d+)", d.get("crs", ""))
    return f"EPSG:{m.group(1)}" if m else None


def raster_crs(ds, utm):
    """Li_карты лежат в LOCAL_CS без EPSG — берём CRS из UTM.json (как read_target_crs_from_utm у конкурента)."""
    try:
        if ds.crs is not None and ds.crs.to_epsg():
            return f"EPSG:{ds.crs.to_epsg()}"
    except Exception:
        pass
    return utm


def geojson_crs(d):
    """CRS geojson-файла: поле crs (в бакете встречается EPSG:3857), иначе RFC7946 → CRS84."""
    n = ((d.get("crs") or {}).get("properties") or {}).get("name", "")
    m = re.search(r"EPSG::?(\d+)", n)
    return f"EPSG:{m.group(1)}" if m else "OGC:CRS84"


def load_marks(paths_by_mod, crs):
    """{модальность: [(shapely геометрия в crs, class_id)]}; невалидные чинятся buffer(0)."""
    from shapely.geometry import shape
    from shapely.ops import transform as stransform
    from pyproj import Transformer
    out, trs = {}, {}
    for path, mod, cls in paths_by_mod:
        d = json.load(open(path, encoding="utf-8")); src = geojson_crs(d)
        if src not in trs: trs[src] = Transformer.from_crs(src, crs, always_xy=True).transform
        for f in d.get("features", []):
            g = f.get("geometry")
            if not g or g.get("type") not in ("Polygon", "MultiPolygon"): continue
            geom = stransform(trs[src], shape(g))
            if not geom.is_valid: geom = geom.buffer(0)
            if geom.is_empty: continue
            out.setdefault(mod, []).append((geom, CLASSES.index(cls)))
    return out


def resize_pair(img, mask, size=256):
    import torch, torch.nn.functional as F
    t = torch.from_numpy(img.astype(np.float32))[None, None]
    im = F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)[0, 0].numpy()
    mk = F.interpolate(torch.from_numpy(mask.astype(np.int64))[None, None].float(), size=(size, size), mode="nearest")[0, 0].long().numpy()
    return im, mk.astype(np.uint8)


def adaptive_patch(ds, geom, objs, min_crop=256, max_crop=4096, ctx=2.0):
    """Копия extract_adaptive_patch_and_multi_mask конкурента (build_multiclass_dataset.py:72–230)."""
    import rasterio
    from affine import Affine
    from rasterio.features import rasterize
    from rasterio.windows import Window
    from shapely.geometry import box
    minx, miny, maxx, maxy = geom.bounds
    pw, ph = abs(ds.transform.a), abs(ds.transform.e)
    ow, oh = (maxx - minx) / pw, (maxy - miny) / ph
    crop = min(max_crop, max(min_crop, int(np.ceil(max(ow, oh) * ctx))))
    crop += crop % 2
    c = geom.centroid
    rc, cc = ds.index(c.x, c.y)
    r0, c0 = rc - crop // 2, cc - crop // 2
    rr0, cc0, rr1, cc1 = max(0, r0), max(0, c0), min(ds.height, r0 + crop), min(ds.width, c0 + crop)
    if rr1 <= rr0 or cc1 <= cc0: return None
    raw = ds.read(1, window=Window(cc0, rr0, cc1 - cc0, rr1 - rr0))
    patch = np.zeros((crop, crop), raw.dtype)
    patch[rr0 - r0:rr0 - r0 + raw.shape[0], cc0 - c0:cc0 - c0 + raw.shape[1]] = raw
    xl, yt = ds.xy(r0, c0, offset="ul")
    tr = Affine(ds.transform.a, ds.transform.b, xl, ds.transform.d, ds.transform.e, yt)
    xr, yb = rasterio.transform.xy(tr, crop, crop, offset="ul")
    win = box(xl, yb, xr, yt)
    items = sorted([(g, v) for g, v in objs if g.intersects(win)], key=lambda t: -t[0].area)
    if not items: return None
    mask = rasterize(items, out_shape=(crop, crop), transform=tr, fill=0, dtype="uint8")
    if not mask.any(): return None
    return patch, mask, crop


def gpu_guard(require=True):
    import torch
    n = torch.cuda.device_count() if torch.cuda.is_available() else 0
    names = [torch.cuda.get_device_name(i) for i in range(n)]
    print("GPU:", names, flush=True)
    if require and n == 0:
        raise SystemExit("--require-gpu: GPU нет (сессия получила CPU — второй одновременный GPU-кернел?). Останов, чтобы не жечь квоту.")
    return n
