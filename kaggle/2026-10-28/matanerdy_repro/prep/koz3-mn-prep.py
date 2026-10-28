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

FOLDS = json.loads(r'''{"source": "make_folds.py (s3/listing.txt, gt_train_v2, bot-10 family_clusters, folds_li_tin_59_t128)", "family_of": {"001_Гнёздово_Нивленское": "Гнёздово", "002_Тешилово": "Тешилово", "003_Лубно": "Лубно", "004_Боровичи_1": "Боровичи", "005_Боровичи_2": "Боровичи", "006_Осечки_1.1км": "Миловидово", "007_Армиево": "Армиево", "009_Хальвасборген": "Хальвасборген", "010_Эсбю_укрепление": "Эсбю", "011_Эрентуна": "Эрентуна", "012_Сож": "Сож", "013_Нора_Виннон": "Нора", "015_Демидовка": "Демидовка", "016_Хога": "Хога", "017_Юшково": "Юшково", "018_Курган_Хога": "Курган", "019_Селяне": "Селяне", "020_Романово": "Романово", "021_Новенькое": "Новенькое", "022_Лихуша": "Лихуша", "023_Бервенец": "Бервенец", "024_Стрекаловка": "Стрекаловка", "025_Вышенское": "Вышенское", "026_Золотаревка": "Золотаревка", "027_Вышегжа_1": "Вышегжа", "028_Вышегжа_2": "Вышегжа", "029_Вышегжа_3": "Вышегжа", "030_Эсбю": "Эсбю", "031_Новотитаровская": "Новотитаровская", "032_Красносельская_0.5км": "Красносельская", "033_Елизаветинское": "Елизаветинское", "034_Усть_река": "Усть", "035_Шумгора": "Шумгора", "036_Тимерево": "Тимерево", "037_Никола_Ленивец": "Никола", "038_Копанское": "Копанское", "039_Примаки_0.7км": "Миловидово", "040_Примаки_1.3км": "Миловидово", "041_Миловидово": "Миловидово", "042_Тору_Айгыр": "Тору", "043_Гочево": "Гочево", "044_Новотитаровская_1.8км": "Новотитаровская", "045_Новотитаровская_2.8км": "Новотитаровская", "046_Тыва_1": "Тыва", "047_Тыва_2": "Тыва", "048_Калмыкия_1": "Калмыкия", "049_Калмыкия_2": "Калмыкия", "050_Воронеж_1": "Воронеж", "051_Воронеж_2": "Воронеж", "052_Воронеж_3": "Воронеж", "053_Воронеж_4": "Воронеж", "054_Воронеж_5": "Воронеж", "055_Курменту": "Курменту", "057_Кабардино_Балкария": "Кабардино", "058_Перынь": "Перынь", "059_Шишкино": "Шишкино", "060_Сельцо": "Сельцо", "061_Граборг": "Граборг", "062_Каменка": "Каменка", "063_Верхнее_Казачье": "Верхнее", "064_Ксизово": "Ксизово", "065_Шниткино": "Шниткино", "066_Московиты": "Московиты", "067_Чудово": "Чудово", "068_Стеклянуха_1": "Стеклянуха", "069_Стеклянуха_2": "Стеклянуха", "070_Архангельское": "Архангельское", "071_Сторожевое": "Сторожевое", "072_Белая_Гора": "Белая", "073_Тоссор": "Тоссор", "074_Солдатское": "Солдатское", "075_Мостище": "Мостище", "076_Маяцкая_крепость": "Маяцкая", "077_Русская_Тростянка": "Русская", "078_Верхняя_Покровка": "Верхняя", "079_Городище": "Городище", "080_Верхний_Карабут": "Верхний", "081_Костомаровское": "Костомаровское", "082_Артпозиции_Иссык_Куль": "Артпозиции", "083_Барскоон": "Барскоон", "084_Булан_Соготту": "Булан", "085_Каджы_Саз": "Иссык", "086_Кан_Дебе": "Кан", "087_Кароол_Дюбе": "Кароол", "088_Кель_Тор_Х": "Кель", "089_Конур_Елен": "Конур", "090_Корумду": "Корумду", "091_Сакские_курганы": "Сакские", "092_Шаты": "Шаты", "094_Иссык_Куль_курганы_и_выкладки": "Иссык", "095_Иссык_Куль_курганы_и_постройки_1": "Иссык", "096_Иссык_Куль_курганы_и_постройки_2": "Иссык", "097_Орнок": "Орнок", "098_Таш_Коро_1": "Таш", "099_Таш_Коро_2": "Таш", "100_Торт_Куль": "Торт", "101_Гнёздово_Днепровская": "Гнёздово", "102_Гнёздово_Заольшанская": "Гнёздово", "103_Гнёздово_Ольшанская": "Гнёздово", "104_Шетнево": "Шетнево", "105_Вщиж": "Вщиж", "106_Верхмарево": "Верхмарево", "107_Сары_Булун": "Сары", "108_Скупая_Полудань": "Скупая", "109_Россошки": "Россошки", "110_Саратов": "Саратов", "111_Каменные_выкладки_1": "Каменные", "112_Каменные_выкладки_2": "Каменные", "113_Каменные_выкладки_3": "Каменные", "114_Каменные_выкладки_4": "Каменные", "115_Каменные_выкладки_5": "Каменные", "116_Каменные_выкладки_6": "Каменные", "117_Каменные_выкладки_7": "Каменные", "118_Каменные_выкладки_8": "Каменные", "119_Каменные_выкладки_9": "Каменные", "120_Каменные_выкладки_10": "Каменные", "121_Каменные_выкладки_11": "Каменные", "122_Каменные_выкладки_12": "Каменные", "123_Каменные_выкладки_13": "Каменные", "124_Постройки_1": "Постройки", "125_Постройки_2": "Постройки", "126_Постройки_3": "Постройки", "127_Постройки_4": "Постройки", "128_Постройки_5": "Постройки", "129_Постройки_6": "Постройки", "130_Курганы_1": "Курганы", "131_Курганы_2": "Курганы", "132_Курганы_3": "Курганы", "133_Курганы_4": "Курганы", "134_Курганы_5": "Курганы", "135_Курганы_6": "Курганы", "136_Курганы_7": "Курганы", "137_Курганы_8": "Курганы", "138_Курганы_9": "Курганы", "139_Курганы_10": "Курганы", "140_Курганы_11": "Курганы", "141_Курганы_12": "Курганы", "142_Курганы_13": "Курганы", "143_Курганы_14": "Курганы", "144_Курганы_15": "Курганы", "145_Курганы_16": "Курганы", "146_Курганы_17": "Курганы", "147_Курганы_18": "Курганы", "148_Курганы_19": "Курганы", "149_Курганы_20": "Курганы", "150_Курганы_21": "Курганы", "151_Курганы_22": "Курганы", "152_Курганы_23": "Курганы", "153_Курганы_24": "Курганы", "154_Курганы_25": "Курганы", "155_Курганы_26": "Курганы", "156_Курганы_27": "Курганы", "157_Курганы_28": "Курганы", "158_Курганы_29": "Курганы", "159_Курганы_30": "Курганы", "160_Курганы_31": "Курганы", "161_Курганы_32": "Курганы", "162_Курганы_33": "Курганы", "163_Курганы_34": "Курганы", "164_Кисловодская_котловина_1": "Кисловодская", "165_Кисловодская_котловина_2": "Кисловодская", "166_Кисловодская_котловина_3": "Кисловодская", "167_Кисловодская_котловина_4": "Кисловодская", "168_Кисловодская_котловина_5": "Кисловодская", "169_Кисловодская_котловина_6": "Кисловодская", "170_Кисловодская_котловина_7": "Кисловодская", "171_Кисловодская_котловина_8": "Кисловодская", "172_Кисловодская_котловина_9": "Кисловодская", "173_Кисловодская_котловина_10": "Кисловодская", "174_Кисловодская_котловина_11": "Кисловодская", "175_Кисловодская_котловина_12": "Кисловодская", "176_Кисловодская_котловина_13": "Кисловодская", "177_Кисловодская_котловина_14": "Кисловодская", "178_Кисловодская_котловина_15": "Кисловодская", "179_Гнёздово_Лесная_1": "Гнёздово", "180_Гнёздово_Лесная_2": "Гнёздово", "181_Гнёздово_Лесная_3": "Гнёздово", "182_Красносельская_1.7км": "Красносельская", "183_Сарское": "Сарское", "184_Шурскол": "Шурскол", "185_Изборск": "Изборск", "186_Семибратнее_1": "Семибратнее", "187_Судьбищи_1": "Судьбищи", "188_Руна": "Руна", "189_Макеево": "Макеево", "190_Муравлево": "Муравлево", "191_Осечки_1.3км": "Миловидово", "192_КЧР": "КЧР", "200_Судьбищи_2": "Судьбищи"}, "folds": [{"fold": 0, "test_families": ["Гнёздово"], "val_family": "Тимерево", "test_regions": ["001_Гнёздово_Нивленское", "101_Гнёздово_Днепровская", "102_Гнёздово_Заольшанская", "103_Гнёздово_Ольшанская", "179_Гнёздово_Лесная_1", "180_Гнёздово_Лесная_2", "181_Гнёздово_Лесная_3"]}, {"fold": 1, "test_families": ["Булан", "Верхняя", "Воронеж", "Кан", "Корумду", "Курменту", "Постройки", "Россошки", "Солдатское", "Сторожевое", "Тимерево", "Тыва", "Шаты", "Шишкино", "Шниткино"], "val_family": "Гнёздово", "test_regions": ["036_Тимерево", "046_Тыва_1", "047_Тыва_2", "050_Воронеж_1", "051_Воронеж_2", "052_Воронеж_3", "053_Воронеж_4", "054_Воронеж_5", "055_Курменту", "059_Шишкино", "065_Шниткино", "071_Сторожевое", "074_Солдатское", "078_Верхняя_Покровка", "084_Булан_Соготту", "086_Кан_Дебе", "090_Корумду", "092_Шаты", "109_Россошки", "124_Постройки_1", "125_Постройки_2", "126_Постройки_3", "127_Постройки_4", "128_Постройки_5", "129_Постройки_6"]}, {"fold": 2, "test_families": ["Артпозиции", "Верхнее", "Вышенское", "Городище", "Демидовка", "Золотаревка", "КЧР", "Кароол", "Костомаровское", "Курганы", "Лихуша", "Московиты", "Никола", "Новотитаровская", "Русская", "Сакские", "Саратов", "Таш", "Торт", "Усть", "Эсбю"], "val_family": "Гнёздово", "test_regions": ["010_Эсбю_укрепление", "015_Демидовка", "022_Лихуша", "025_Вышенское", "026_Золотаревка", "030_Эсбю", "031_Новотитаровская", "034_Усть_река", "037_Никола_Ленивец", "044_Новотитаровская_1.8км", "045_Новотитаровская_2.8км", "063_Верхнее_Казачье", "066_Московиты", "077_Русская_Тростянка", "079_Городище", "081_Костомаровское", "082_Артпозиции_Иссык_Куль", "087_Кароол_Дюбе", "091_Сакские_курганы", "098_Таш_Коро_1", "099_Таш_Коро_2", "100_Торт_Куль", "110_Саратов", "130_Курганы_1", "131_Курганы_2", "132_Курганы_3", "133_Курганы_4", "134_Курганы_5", "135_Курганы_6", "136_Курганы_7", "137_Курганы_8", "138_Курганы_9", "139_Курганы_10", "140_Курганы_11", "141_Курганы_12", "142_Курганы_13", "143_Курганы_14", "144_Курганы_15", "145_Курганы_16", "146_Курганы_17", "147_Курганы_18", "148_Курганы_19", "149_Курганы_20", "150_Курганы_21", "151_Курганы_22", "152_Курганы_23", "153_Курганы_24", "154_Курганы_25", "155_Курганы_26", "156_Курганы_27", "157_Курганы_28", "158_Курганы_29", "159_Курганы_30", "160_Курганы_31", "161_Курганы_32", "162_Курганы_33", "163_Курганы_34", "192_КЧР"]}, {"fold": 3, "test_families": ["Бервенец", "Боровичи", "Верхмарево", "Вщиж", "Гочево", "Елизаветинское", "Иссык", "Кель", "Ксизово", "Макеево", "Миловидово", "Новенькое", "Нора", "Орнок", "Руна", "Сарское", "Сельцо", "Семибратнее", "Скупая", "Тешилово", "Тору", "Чудово", "Шетнево", "Шурскол"], "val_family": "Гнёздово", "test_regions": ["002_Тешилово", "004_Боровичи_1", "005_Боровичи_2", "006_Осечки_1.1км", "013_Нора_Виннон", "021_Новенькое", "023_Бервенец", "033_Елизаветинское", "039_Примаки_0.7км", "040_Примаки_1.3км", "041_Миловидово", "042_Тору_Айгыр", "043_Гочево", "060_Сельцо", "064_Ксизово", "067_Чудово", "085_Каджы_Саз", "088_Кель_Тор_Х", "094_Иссык_Куль_курганы_и_выкладки", "095_Иссык_Куль_курганы_и_постройки_1", "096_Иссык_Куль_курганы_и_постройки_2", "097_Орнок", "104_Шетнево", "105_Вщиж", "106_Верхмарево", "108_Скупая_Полудань", "183_Сарское", "184_Шурскол", "186_Семибратнее_1", "188_Руна", "189_Макеево", "191_Осечки_1.3км"]}, {"fold": 4, "test_families": ["Армиево", "Архангельское", "Барскоон", "Белая", "Верхний", "Вышегжа", "Граборг", "Изборск", "Кабардино", "Калмыкия", "Каменка", "Каменные", "Кисловодская", "Конур", "Копанское", "Красносельская", "Курган", "Лубно", "Маяцкая", "Мостище", "Муравлево", "Перынь", "Романово", "Сары", "Селяне", "Сож", "Стеклянуха", "Стрекаловка", "Судьбищи", "Тоссор", "Хальвасборген", "Хога", "Шумгора", "Эрентуна", "Юшково"], "val_family": "Гнёздово", "test_regions": ["003_Лубно", "007_Армиево", "009_Хальвасборген", "011_Эрентуна", "012_Сож", "016_Хога", "017_Юшково", "018_Курган_Хога", "019_Селяне", "020_Романово", "024_Стрекаловка", "027_Вышегжа_1", "028_Вышегжа_2", "029_Вышегжа_3", "032_Красносельская_0.5км", "035_Шумгора", "038_Копанское", "048_Калмыкия_1", "049_Калмыкия_2", "057_Кабардино_Балкария", "058_Перынь", "061_Граборг", "062_Каменка", "068_Стеклянуха_1", "069_Стеклянуха_2", "070_Архангельское", "072_Белая_Гора", "073_Тоссор", "075_Мостище", "076_Маяцкая_крепость", "080_Верхний_Карабут", "083_Барскоон", "089_Конур_Елен", "107_Сары_Булун", "111_Каменные_выкладки_1", "112_Каменные_выкладки_2", "113_Каменные_выкладки_3", "114_Каменные_выкладки_4", "115_Каменные_выкладки_5", "116_Каменные_выкладки_6", "117_Каменные_выкладки_7", "118_Каменные_выкладки_8", "119_Каменные_выкладки_9", "120_Каменные_выкладки_10", "121_Каменные_выкладки_11", "122_Каменные_выкладки_12", "123_Каменные_выкладки_13", "164_Кисловодская_котловина_1", "165_Кисловодская_котловина_2", "166_Кисловодская_котловина_3", "167_Кисловодская_котловина_4", "168_Кисловодская_котловина_5", "169_Кисловодская_котловина_6", "170_Кисловодская_котловина_7", "171_Кисловодская_котловина_8", "172_Кисловодская_котловина_9", "173_Кисловодская_котловина_10", "174_Кисловодская_котловина_11", "175_Кисловодская_котловина_12", "176_Кисловодская_котловина_13", "177_Кисловодская_котловина_14", "178_Кисловодская_котловина_15", "182_Красносельская_1.7км", "185_Изборск", "187_Судьбищи_1", "190_Муравлево", "200_Судьбищи_2"]}]}''')

# ==== Ядро 1 (CPU, интернет): S3 → обучающие патчи по рецепту MataNerdy + манифест растров для инференса ====
import sys, csv, shutil, concurrent.futures as cf
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
OUT = Path(os.environ.get("MN_OUT", "/kaggle/working"))


def prep_region(cli, region, keys):
    import rasterio
    rast, marks, utm = region_files(keys)
    if utm is None: return [], None, {"region": region, "skip": "нет UTM.json"}
    crs_utm = utm_crs(fetch(cli, *utm))
    mpaths = [(fetch(cli, k, s), mod, cls) for k, s, mod, cls in marks]
    rows, X, M = [], [], []
    info = {"region": region, "utm": crs_utm, "rasters": {}}
    for mod, lst in rast.items():
        mods_marks = [(p, m, c) for p, m, c in mpaths if m == mod]
        info["rasters"][mod] = [{"key": k, "size": s} for k, s in lst]
        if not lst or not mods_marks: continue
        opened = []
        for k, s in lst:
            ds = rasterio.open(fetch(cli, k, s)); opened.append((k, ds, raster_crs(ds, crs_utm)))
        for k, ds, crs in opened:
            info["rasters"][mod][[r["key"] for r in info["rasters"][mod]].index(k)]["crs"] = crs
        by_crs = {}
        for k, ds, crs in opened:
            if crs not in by_crs: by_crs[crs] = load_marks(mods_marks, crs).get(mod, [])
        done = set()
        for k, ds, crs in opened:  # первый растр, с которым пересекается объект (как у конкурента)
            from shapely.geometry import box
            bb = box(*ds.bounds)
            objs = by_crs[crs]
            for i, (g, v) in enumerate(objs):
                if i in done or not g.intersects(bb): continue
                done.add(i)
                r = adaptive_patch(ds, g, objs)
                if r is None: continue
                patch, mask, crop = r
                im, mk = resize_pair(patch, mask)
                X.append(im.astype(np.float16)); M.append(mk)
                rows.append({"region": region, "modality": mod, "markup_type": MOD2MARKUP[mod], "raster": k,
                             "class_name": CLASSES[v], "crop_size": crop})
        for _, ds, _ in opened: ds.close()
    if os.environ.get("MN_KEEP", "0") != "1":
        shutil.rmtree(CACHE / region, ignore_errors=True)
    return rows, (np.stack(X), np.stack(M)) if X else None, info


def main():
    t0 = time.time(); cli = s3_client()
    regions = sorted(FOLDS["family_of"])
    if os.environ.get("MN_REGIONS"): regions = [r for r in regions if r in os.environ["MN_REGIONS"].split(",")]
    fold_of = {r: f["fold"] for f in FOLDS["folds"] for r in f["test_regions"]}
    allrows, XS, MS, manifest = [], [], [], {}
    with cf.ThreadPoolExecutor(3) as ex:  # 3 участка параллельно: скачивание перекрывается с нарезкой
        futs = {ex.submit(lambda r: prep_region(cli, r, list_region(cli, r)), r): r for r in regions}
        for fu in cf.as_completed(futs):
            r = futs[fu]
            try:
                rows, xm, info = fu.result()
            except Exception as e:
                log.error("участок %s: %s", r, e); manifest[r] = {"error": str(e)[:300]}; continue
            for row in rows: row.update(family=FOLDS["family_of"][r], fold=fold_of[r])
            allrows += rows; manifest[r] = info
            if xm is not None: XS.append(xm[0]); MS.append(xm[1])
            log.info("%s: %d патчей, всего %d, %.0f с", r, len(rows), len(allrows), time.time() - t0)
    X = np.concatenate(XS) if XS else np.zeros((0, 256, 256), np.float16)
    M = np.concatenate(MS) if MS else np.zeros((0, 256, 256), np.uint8)
    OUT.mkdir(parents=True, exist_ok=True)
    np.save(OUT / "X.npy", X); np.save(OUT / "M.npy", M)
    with open(OUT / "meta.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["idx", "region", "family", "fold", "modality", "markup_type", "raster", "class_name", "crop_size"])
        w.writeheader()
        for i, row in enumerate(allrows): w.writerow({"idx": i, **row})
    json.dump({"folds": FOLDS, "regions": manifest}, open(OUT / "manifest.json", "w", encoding="utf-8"), ensure_ascii=False)
    import collections
    summ = {"patches": len(allrows), "sec": round(time.time() - t0),
            "by_mod_class": collections.Counter(f'{r["modality"]}/{r["class_name"]}' for r in allrows),
            "by_fold": collections.Counter(r["fold"] for r in allrows), "errors": [r for r, v in manifest.items() if "error" in v]}
    json.dump(summ, open(OUT / "prep_summary.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(summ, ensure_ascii=False))


if __name__ == "__main__":
    main()
